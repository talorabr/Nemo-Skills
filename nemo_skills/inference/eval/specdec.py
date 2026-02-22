# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import glob
import json
import logging
import os
import shutil
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import hydra
import requests

from nemo_skills.inference.generate import (
    GenerationTask,
    GenerationTaskConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


# ---------------------------------------------------------------------------
# Speculative decoding metrics from the server's Prometheus endpoint
# ---------------------------------------------------------------------------


@dataclass
class SpecDecodeMetrics:
    """Raw speculative decoding counters scraped from the Prometheus endpoint."""

    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    accepted_per_pos: dict[int, int] = field(default_factory=dict)


def fetch_spec_decode_metrics(base_url: str) -> SpecDecodeMetrics | None:
    """Fetch speculative decoding metrics from the server's ``/metrics`` endpoint.

    Parses Prometheus text-format exposition looking for VLLM
    ``vllm:spec_decode_*`` counters including per-position acceptance counts.

    Args:
        base_url: Server root URL, e.g. ``http://127.0.0.1:5000``.

    Returns:
        :class:`SpecDecodeMetrics` with the scraped counters, or ``None``
        if speculative decoding metrics are not available.
    """
    metrics_url = f"{base_url.rstrip('/')}/metrics"
    try:
        response = requests.get(metrics_url, timeout=30)
        if response.status_code != 200:
            LOG.warning("Metrics endpoint returned status %d", response.status_code)
            return None
        text = response.text

        num_drafts = 0
        num_draft_tokens = 0
        num_accepted_tokens = 0
        accepted_per_pos: dict[int, int] = {}
        found_spec_decode = False

        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("vllm:spec_decode"):
                found_spec_decode = True
                # Skip Prometheus _created timestamp lines — they contain
                # Unix timestamps, not actual counter values.
                if "_created" in line:
                    continue
                parts = line.split()
                if parts:
                    with contextlib.suppress(ValueError):
                        if "num_drafts" in line:
                            num_drafts += int(float(parts[-1]))
                        elif "num_draft_tokens" in line:
                            num_draft_tokens += int(float(parts[-1]))
                        elif "num_accepted_tokens_per_pos" in line:
                            pos_label = 'position="'
                            if pos_label in line:
                                start = line.index(pos_label) + len(pos_label)
                                end = line.index('"', start)
                                pos = int(line[start:end])
                                val = int(float(parts[-1]))
                                accepted_per_pos[pos] += val
                        elif "num_accepted_tokens" in line:
                            num_accepted_tokens += int(float(parts[-1]))

        if not found_spec_decode:
            LOG.info("No vllm:spec_decode_* metrics found on the server (speculative decoding may not be enabled).")
            return None

        return SpecDecodeMetrics(
            num_drafts=num_drafts,
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens,
            accepted_per_pos=accepted_per_pos,
        )
    except requests.RequestException as exc:
        LOG.warning("Failed to fetch metrics from %s: %s", metrics_url, exc)
        return None


def read_sglang_metrics_file(
    metrics_file_path: str,
    request_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Read SGLang metrics file and extract speculative decoding metrics.

    SGLang writes per-request metrics as JSON lines when launched with
    ``--export-metrics-to-file``.  Each line contains:

    - ``id``: Request ID (matches the ``rid`` in request_parameters)
    - ``spec_accept_length``: Acceptance length for this request
    - ``spec_accept_rate``: Acceptance rate for this request
    - ``spec_accept_token_num``: Number of accepted tokens
    - ``spec_draft_token_num``: Number of draft tokens
    - ``spec_verify_ct``: Number of verification steps (drafts)

    Args:
        metrics_file_path: Path to the SGLang metrics JSONL file.
        request_ids: Optional set of request IDs to filter by.  If ``None``,
            **all** entries in the file are aggregated (useful when the
            server is dedicated to the benchmark).

    Returns:
        Dictionary with aggregated spec decode statistics, or ``None`` if no
        matching requests were found or metrics are unavailable.
    """
    if not os.path.exists(metrics_file_path):
        LOG.warning("SGLang metrics file not found: %s", metrics_file_path)
        return None

    matching_entries = []
    total_drafts = 0
    total_draft_tokens = 0
    total_accepted_tokens = 0

    try:
        with open(metrics_file_path, "rt", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # If request_ids is None, accept all entries
                    if request_ids is not None:
                        req_id = entry["id"]
                        if req_id not in request_ids:
                            continue
                    matching_entries.append(entry)
                    # SGLang uses spec_verify_ct as the draft count
                    total_drafts += entry["spec_verify_ct"]
                    total_draft_tokens += entry["spec_draft_token_num"]
                    total_accepted_tokens += entry["spec_accept_token_num"]
                except json.JSONDecodeError:
                    LOG.warning("Skipping malformed JSON line in metrics file: %s", line[:100])
                    continue

        if not matching_entries:
            filter_desc = f"matching {len(request_ids)} IDs" if request_ids else "in file"
            LOG.warning("No entries found %s in SGLang metrics file.", filter_desc)
            return None

        if total_draft_tokens <= 0:
            LOG.warning(
                "No speculative decoding activity detected (total_draft_tokens=%d). "
                "Metrics will be empty.",
                total_draft_tokens,
            )
            return None

        # Compute aggregate metrics
        acceptance_rate = (total_accepted_tokens / total_draft_tokens) * 100 if total_draft_tokens > 0 else 0.0
        acceptance_length = 1 + (total_accepted_tokens / total_drafts) if total_drafts > 0 else 0.0

        per_pos_rates: list[float] = []  # SGLang does not expose per-position breakdown

        LOG.info(
            "SGLang metrics file: entries=%d, drafts=%d, draft_tokens=%d, "
            "accepted=%d, acceptance_rate=%.2f%%, acceptance_length=%.4f",
            len(matching_entries),
            total_drafts,
            total_draft_tokens,
            total_accepted_tokens,
            acceptance_rate,
            acceptance_length,
        )

        return {
            "num_drafts": total_drafts,
            "draft_tokens": total_draft_tokens,
            "accepted_tokens": total_accepted_tokens,
            "acceptance_rate": acceptance_rate,
            "acceptance_length": acceptance_length,
            "per_position_acceptance_rates": per_pos_rates,
        }
    except Exception as exc:
        LOG.warning("Failed to read SGLang metrics file %s: %s", metrics_file_path, exc)
        return None


def find_sglang_metrics_file(metrics_dir: str) -> str | None:
    """Find the most recent SGLang metrics file in the given directory.

    SGLang creates files like ``sglang-request-metrics-YYYYMMDD_HH.log``.

    Args:
        metrics_dir: Directory containing SGLang metrics files.

    Returns:
        Path to the most recent metrics file, or ``None`` if not found.
    """
    if not os.path.isdir(metrics_dir):
        LOG.warning("SGLang metrics directory does not exist: %s", metrics_dir)
        return None

    pattern = os.path.join(metrics_dir, "sglang-request-metrics-*.log")
    files = glob.glob(pattern)
    if not files:
        LOG.warning("No SGLang metrics files found matching pattern: %s", pattern)
        return None

    # Return the most recently modified file
    latest_file = max(files, key=os.path.getmtime)
    LOG.info("Using SGLang metrics file: %s", latest_file)
    return latest_file


# ---------------------------------------------------------------------------
# SGLang Prometheus metrics (before/after delta on the /metrics endpoint)
# ---------------------------------------------------------------------------


@dataclass
class SglangSpecDecodeMetrics:
    """Snapshot of SGLang speculative decoding metrics from the Prometheus endpoint.

    SGLang exposes two *gauge* metrics (instantaneous averages over the most
    recent scheduler batch) and several *counter* metrics that monotonically
    increase.  We capture both so we can reconstruct per-benchmark averages
    using the counter deltas as weights.

    Prometheus lines of interest (from ``/metrics``)::

        sglang:spec_accept_length{…}   <float>   # gauge
        sglang:spec_accept_rate{…}     <float>   # gauge
        sglang:num_requests_total{…}   <float>   # counter
        sglang:generation_tokens_total{…} <float> # counter
    """

    spec_accept_length: float = 0.0
    spec_accept_rate: float = 0.0
    num_requests: int = 0
    generation_tokens: int = 0


def fetch_sglang_spec_decode_metrics(base_url: str) -> SglangSpecDecodeMetrics | None:
    """Fetch speculative decoding metrics from an SGLang server's ``/metrics`` endpoint.

    Parses Prometheus text-format exposition looking for ``sglang:spec_accept_*``
    gauges and ``sglang:num_requests_total`` / ``sglang:generation_tokens_total``
    counters.

    Args:
        base_url: Server root URL, e.g. ``http://127.0.0.1:5000``.

    Returns:
        :class:`SglangSpecDecodeMetrics` with the scraped values, or ``None``
        if the metrics endpoint is unreachable or speculative decoding gauges
        are absent.
    """
    metrics_url = f"{base_url.rstrip('/')}/metrics"
    try:
        response = requests.get(metrics_url, timeout=30)
        if response.status_code != 200:
            LOG.warning("SGLang metrics endpoint returned status %d", response.status_code)
            return None
        text = response.text

        spec_accept_length = 0.0
        spec_accept_rate = 0.0
        num_requests = 0
        generation_tokens = 0
        found_spec = False

        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            with contextlib.suppress(ValueError):
                # Gauges – SGLang may emit these with labels; we accumulate
                # across TP ranks (though typically only tp_rank=0 carries
                # spec decode info).
                if "sglang:spec_accept_length{" in line or line.startswith("sglang:spec_accept_length "):
                    spec_accept_length = float(parts[-1])
                    found_spec = True
                elif "sglang:spec_accept_rate{" in line or line.startswith("sglang:spec_accept_rate "):
                    spec_accept_rate = float(parts[-1])
                    found_spec = True
                # Counters
                elif "sglang:num_requests_total{" in line or line.startswith("sglang:num_requests_total "):
                    num_requests = int(float(parts[-1]))
                elif "sglang:generation_tokens_total{" in line or line.startswith("sglang:generation_tokens_total "):
                    generation_tokens = int(float(parts[-1]))

        if not found_spec:
            LOG.info(
                "No sglang:spec_accept_* metrics found on the server "
                "(speculative decoding may not be enabled)."
            )
            return None

        return SglangSpecDecodeMetrics(
            spec_accept_length=spec_accept_length,
            spec_accept_rate=spec_accept_rate,
            num_requests=num_requests,
            generation_tokens=generation_tokens,
        )
    except requests.RequestException as exc:
        LOG.warning("Failed to fetch SGLang metrics from %s: %s", metrics_url, exc)
        return None


def compute_sglang_spec_decode_delta(
    before: SglangSpecDecodeMetrics,
    after: SglangSpecDecodeMetrics,
) -> dict[str, Any] | None:
    """Compute benchmark-specific acceptance metrics from two SGLang snapshots.

    SGLang exposes ``spec_accept_length`` and ``spec_accept_rate`` as **gauges**
    (running averages from the server's perspective).  Combined with the
    ``num_requests_total`` and ``generation_tokens_total`` **counters** we can
    back out the benchmark-specific averages::

        weighted_al_after  = al_after  x n_after
        weighted_al_before = al_before x n_before
        benchmark_al = (weighted_al_after - weighted_al_before)
                       / (n_after - n_before)

    If the before snapshot had 0 requests (fresh server), the after values are
    returned as-is.

    Args:
        before: Snapshot taken before generation.
        after: Snapshot taken after generation.

    Returns:
        Dictionary with ``acceptance_length``, ``acceptance_rate``, counters,
        etc., or ``None`` if no requests were generated.
    """
    delta_requests = after.num_requests - before.num_requests
    delta_gen_tokens = after.generation_tokens - before.generation_tokens

    if delta_requests <= 0:
        LOG.warning(
            "SGLang: no new requests between before (%d) and after (%d) snapshots.",
            before.num_requests,
            after.num_requests,
        )
        return None

    # --- Acceptance length ---
    if before.num_requests == 0:
        # Fresh server — after values cover exactly our benchmark traffic.
        acceptance_length = after.spec_accept_length
    else:
        # Weighted-average decomposition
        weighted_after = after.spec_accept_length * after.num_requests
        weighted_before = before.spec_accept_length * before.num_requests
        acceptance_length = (weighted_after - weighted_before) / delta_requests

    # --- Acceptance rate ---
    if before.num_requests == 0:
        acceptance_rate_fraction = after.spec_accept_rate
    else:
        weighted_after = after.spec_accept_rate * after.num_requests
        weighted_before = before.spec_accept_rate * before.num_requests
        acceptance_rate_fraction = (weighted_after - weighted_before) / delta_requests

    acceptance_rate_pct = acceptance_rate_fraction * 100

    LOG.info(
        "SGLang Prometheus delta: requests=%d, gen_tokens=%d, "
        "acceptance_length=%.4f, acceptance_rate=%.2f%%",
        delta_requests,
        delta_gen_tokens,
        acceptance_length,
        acceptance_rate_pct,
    )

    return {
        "num_drafts": delta_requests,  # one verify cycle per request as proxy
        "draft_tokens": delta_gen_tokens,
        "accepted_tokens": int(delta_gen_tokens * acceptance_rate_fraction) if delta_gen_tokens > 0 else 0,
        "acceptance_rate": acceptance_rate_pct,
        "acceptance_length": acceptance_length,
        "per_position_acceptance_rates": [],  # SGLang does not expose per-position breakdown
    }


def compute_spec_decode_delta(
    before: SpecDecodeMetrics,
    after: SpecDecodeMetrics,
) -> dict[str, Any] | None:
    """Compute the delta of speculative decoding counters between two snapshots.

    Uses the before/after pattern to isolate metrics to only the generation
    run (since Prometheus counters are cumulative).

    Computes:
        * **acceptance_rate** — ``delta_accepted / delta_draft_tokens * 100``
        * **acceptance_length** — ``1 + delta_accepted / delta_drafts``
          (the "+1" accounts for the target model's verified token that
          is always emitted even on a full rejection)
        * **per_position_acceptance_rates** — per-position acceptance
          probability, computed as ``delta_per_pos[i] / delta_drafts``

    Args:
        before: Counters snapshot taken before generation.
        after: Counters snapshot taken after generation.

    Returns:
        Dictionary with computed spec decode statistics, or ``None`` if the
        delta contains no meaningful data.
    """
    delta_drafts = after.num_drafts - before.num_drafts
    delta_draft_tokens = after.num_draft_tokens - before.num_draft_tokens
    delta_accepted = after.num_accepted_tokens - before.num_accepted_tokens

    # Compute per-position acceptance rates
    per_pos_rates: list[float] = []
    if delta_drafts > 0:
        positions = sorted(
            set(before.accepted_per_pos.keys()) | set(after.accepted_per_pos.keys())
        )
        for pos in positions:
            before_val = before.accepted_per_pos[pos]
            after_val = after.accepted_per_pos[pos]
            delta_pos = after_val - before_val
            per_pos_rates.append(delta_pos / delta_drafts)

    if delta_draft_tokens <= 0:
        LOG.warning(
            "No speculative decoding activity detected during generation "
            "(delta_draft_tokens=%d). Metrics will be empty.",
            delta_draft_tokens,
        )
        return None

    acceptance_rate = (delta_accepted / delta_draft_tokens) * 100
    acceptance_length = 1 + delta_accepted / delta_drafts if delta_drafts > 0 else 0.0

    return {
        "num_drafts": delta_drafts,
        "draft_tokens": delta_draft_tokens,
        "accepted_tokens": delta_accepted,
        "acceptance_rate": acceptance_rate,
        "acceptance_length": acceptance_length,
        "per_position_acceptance_rates": per_pos_rates,
    }


# ---------------------------------------------------------------------------
# Generation config & task
# ---------------------------------------------------------------------------


@nested_dataclass(kw_only=True)
class SpecdecGenerationConfig(GenerationTaskConfig):
    """SPEED-Bench generation config for speculative decoding evaluation.

    Extends the standard generation config to inject server connection
    information into the evaluator so it can query the server's ``/metrics``
    Prometheus endpoint after generation completes.

    For the full list of supported parameters, use
    ``python -m nemo_skills.inference.generate --help``
    """

    # Directory to write SGLang metrics to.  If not specified, will use a
    # temporary directory created at launch.  This is the only reliable way
    # to share the tempdir between the pipeline process (which builds the
    # server command) and this generation worker process.
    metrics_file_dir: str | None = None
    max_concurrent_requests: int = 32

    def _post_init_validate_server(self):
        super()._post_init_validate_server()
        assert self.server["server_type"] in ["sglang", "vllm"], f"server_type must be either 'sglang' or 'vllm' for specdec generation, got '{self.server['server_type']}'"

cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_specdec_generation_config", node=SpecdecGenerationConfig)


class SpecdecGenerationTask(GenerationTask):
    """Custom generation task for SPEED-Bench speculative decoding evaluation.

    Captures speculative decoding counters from the server's ``/metrics``
    Prometheus endpoint *before* and *after* generation, then computes the
    delta to derive accurate acceptance length (AL), acceptance rate, and
    per-position conditional acceptance rates.
    """

    _sglang_metrics_dir: str | None = None

    def __init__(self, cfg: SpecdecGenerationConfig):
        super().__init__(cfg)
        self._before_spec_metrics: SpecDecodeMetrics | None = None
        self._before_sglang_metrics: SglangSpecDecodeMetrics | None = None
        self._request_ids: list[str] = []  # Track request IDs for SGLang metrics matching

    @classmethod
    def _ensure_sglang_metrics_dir(cls) -> str:
        """Return (and lazily create) a unique temp directory for SGLang metrics."""
        if cls._sglang_metrics_dir is None:
            import tempfile

            cls._sglang_metrics_dir = tempfile.mkdtemp(prefix="sglang-metrics-")
            LOG.info("Created SGLang metrics temp directory: %s", cls._sglang_metrics_dir)
        return cls._sglang_metrics_dir

    @classmethod
    def get_generation_default_args(cls) -> str:
        """Pass the SGLang metrics temp-directory to the generation worker.

        Called in the **pipeline** process.  The returned hydra override is
        appended to the generation command, so the worker can read the same
        path from ``self.cfg.server.metrics_file_dir``.
        """
        metrics_dir = cls._ensure_sglang_metrics_dir()
        return f"++metrics_file_dir={metrics_dir}"

    @classmethod
    def get_server_command_fn(cls) -> callable:
        """Return a wrapper around the default server command builder.

        When the server type is ``sglang``, the wrapper automatically appends
        ``--enable-metrics --export-metrics-to-file --export-metrics-to-file-dir``
        so that SGLang writes per-request speculative decoding metrics that the
        evaluator can aggregate after generation.
        """
        from nemo_skills.pipeline.utils import get_server_command

        metrics_dir = cls._ensure_sglang_metrics_dir()
        sglang_metrics_args = (
            f"--enable-metrics "
            f"--export-metrics-to-file "
            f"--export-metrics-to-file-dir {metrics_dir}"
        )

        def specdec_server_command(
            server_type,
            num_gpus,
            num_nodes,
            model_path,
            cluster_config,
            server_port,
            server_args="",
            server_entrypoint=None,
        ):
            if server_type == "sglang":
                server_args = f"{server_args} {sglang_metrics_args}".strip()
            return get_server_command(
                server_type=server_type,
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                model_path=model_path,
                cluster_config=cluster_config,
                server_port=server_port,
                server_args=server_args,
                server_entrypoint=server_entrypoint,
            )

        return specdec_server_command

    def fill_prompt(self, data_point, data):
        """Map SPEED-Bench ``turns`` field to ``question`` for prompt filling.

        SPEED-Bench data uses a ``turns`` list (following MT-bench format),
        but the generic/default prompt config expects a ``{question}`` field.
        This method copies the first turn into ``question`` before delegating
        to the parent implementation.  For multi-turn data points, use
        :meth:`fill_prompt_for_turn` instead.
        """
        data_point = deepcopy(data_point)
        if "question" not in data_point and "turns" in data_point:
            turns = data_point["turns"]
            if isinstance(turns, list) and len(turns) > 0:
                data_point["question"] = turns[0]
            else:
                data_point["question"] = str(turns)
        return super().fill_prompt(data_point, data)

    def fill_prompt_for_turn(self, data_point, data, turn_idx, conversation):
        """Build the prompt for a specific turn of a multi-turn conversation.

        For turn 0, this delegates to :meth:`fill_prompt`.  For subsequent
        turns, it appends the assistant response from the previous turn and
        the new user turn to the existing conversation.

        Args:
            data_point: The original data point dictionary.
            data: Full dataset (passed through to parent).
            turn_idx: The 0-based turn index.
            conversation: The accumulated conversation (list of message dicts).

        Returns:
            The updated conversation as a list of message dicts.
        """
        turns = data_point["turns"]
        if turn_idx == 0:
            return self.fill_prompt(data_point, data)
        # Append the new user turn to the existing conversation
        conversation = list(conversation)
        conversation.append({"role": "user", "content": turns[turn_idx]})
        return conversation

    async def process_single_datapoint(self, data_point, all_data):
        """Handle single-turn and multi-turn generation.

        For single-turn data points (or when ``multiturn`` is ``False``), this
        falls through to the base-class implementation.

        For multi-turn data points, it loops through every turn, building up
        the conversation and accumulating token counts.

        Also tracks request IDs for SGLang metrics matching.
        """
        server_type = self.cfg.server["server_type"]
        is_sglang = server_type == "sglang"

        turns = data_point["turns"]
        is_multiturn = data_point["multiturn"] and len(turns) > 1

        if not is_multiturn:
            # For SGLang, we need include_response to extract request ID
            # For VLLM, use the base implementation
            if is_sglang:
                # Replicate base class logic but with include_response=True
                if is_dataclass(self.cfg.inference):
                    inference_params = asdict(self.cfg.inference)
                else:
                    inference_params = dict(self.cfg.inference)
                inference_params["include_response"] = True

                generation_params = {
                    **inference_params,
                    **self.extra_generate_params,
                    "prompt": self.fill_prompt(data_point, all_data),
                    "stop_phrases": [self.cfg.stop_phrase] if self.cfg.stop_phrase else None,
                }

                result = await self.generate_with_semaphore(**generation_params)

                # Handle count_prompt_tokens (same as base class)
                if self.cfg.count_prompt_tokens:
                    from nemo_skills.prompt.utils import get_token_count
                    num_input_tokens = get_token_count(self.hf_tokenizer, generation_params["prompt"])
                    result["num_input_tokens"] = num_input_tokens

                # Extract request ID for SGLang, then remove response object (not JSON serializable)
                if "response" in result:
                    try:
                        response_obj = result["response"]
                        # OpenAI API response objects have .id attribute
                        req_id = getattr(response_obj, "id", None)
                        if req_id:
                            self._request_ids.append(req_id)
                            LOG.debug("Captured SGLang request ID: %s", req_id)
                    except Exception as exc:
                        LOG.warning("Failed to extract request ID from response: %s", exc)
                    finally:
                        # Remove response object to avoid JSON serialization errors
                        result.pop("response", None)

                return result
            else:
                return await super().process_single_datapoint(data_point, all_data)

        # --- Multi-turn generation ---
        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            inference_params = dict(self.cfg.inference)

        # For SGLang, include response to extract request IDs
        if is_sglang:
            inference_params["include_response"] = True

        conversation = None
        total_generated_tokens = 0
        all_generations = []

        for turn_idx in range(len(turns)):
            conversation = self.fill_prompt_for_turn(data_point, all_data, turn_idx, conversation)

            generation_params = {
                **inference_params,
                **self.extra_generate_params,
                "prompt": conversation,
                "stop_phrases": [self.cfg.stop_phrase] if self.cfg.stop_phrase else None,
            }

            result = await self.generate_with_semaphore(**generation_params)
            generation_text = result["generation"]
            num_tokens = result["num_generated_tokens"]

            # Extract request ID for SGLang, then remove response object (not JSON serializable)
            if is_sglang and "response" in result:
                try:
                    response_obj = result["response"]
                    # OpenAI API response objects have .id attribute
                    req_id = getattr(response_obj, "id", None)
                    if req_id:
                        self._request_ids.append(req_id)
                        LOG.debug("Captured SGLang request ID: %s", req_id)
                except Exception as exc:
                    LOG.warning("Failed to extract request ID from response: %s", exc)
                finally:
                    # Remove response object to avoid JSON serialization errors
                    result.pop("response", None)

            all_generations.append(generation_text)
            total_generated_tokens += num_tokens

            # Append assistant response to the conversation for the next turn
            conversation = list(conversation)
            conversation.append({"role": "assistant", "content": generation_text})

        # Return aggregated results
        return {
            "generation": all_generations,
            "num_generated_tokens": total_generated_tokens,
        }

    def _get_server_base_address(self) -> str:
        """Derive the server base address (without ``/v1``) from the config.

        Returns:
            Server base address string, e.g. ``http://127.0.0.1:5000``.
        """
        server_cfg = self.cfg.server
        host = server_cfg["host"]
        port = server_cfg["port"]
        return f"http://{host}:{port}"

    def wait_for_server(self):
        """Wait for the server, then snapshot speculative decoding counters.

        For VLLM: Captures the "before" VLLM Prometheus counters
        (``vllm:spec_decode_*``) so that after generation we can compute a
        clean delta isolated to our generation run.

        For SGLang: Captures the "before" SGLang Prometheus gauges
        (``sglang:spec_accept_length``, ``sglang:spec_accept_rate``) and
        counters (``sglang:num_requests_total``, etc.) for the same purpose.
        If the metrics file (``--export-metrics-to-file``) is available after
        generation, it takes priority over the Prometheus delta.
        """
        super().wait_for_server()
        server_type = self.cfg.server["server_type"]
        base_url = self._get_server_base_address()

        if server_type == "sglang":
            # SGLang: Fetch "before" snapshot from Prometheus /metrics
            LOG.info("Fetching BEFORE SGLang spec-decode metrics from %s/metrics", base_url)
            self._before_sglang_metrics = fetch_sglang_spec_decode_metrics(base_url)
            if self._before_sglang_metrics is not None:
                LOG.info(
                    "SGLang before snapshot: accept_length=%.4f, accept_rate=%.4f, "
                    "num_requests=%d, gen_tokens=%d",
                    self._before_sglang_metrics.spec_accept_length,
                    self._before_sglang_metrics.spec_accept_rate,
                    self._before_sglang_metrics.num_requests,
                    self._before_sglang_metrics.generation_tokens,
                )
            else:
                LOG.info(
                    "No SGLang spec-decode metrics found before generation "
                    "(speculative decoding may not be enabled)."
                )
            return

        # VLLM: Fetch before snapshot
        LOG.info("Fetching BEFORE spec-decode metrics from %s/metrics", base_url)
        self._before_spec_metrics = fetch_spec_decode_metrics(base_url)
        if self._before_spec_metrics is not None:
            LOG.info(
                "Before snapshot: drafts=%d, draft_tokens=%d, accepted=%d, per_pos_keys=%s",
                self._before_spec_metrics.num_drafts,
                self._before_spec_metrics.num_draft_tokens,
                self._before_spec_metrics.num_accepted_tokens,
                sorted(self._before_spec_metrics.accepted_per_pos.keys()),
            )
        else:
            LOG.info("No spec-decode metrics found before generation (may not be enabled).")

    def run_batch_evaluation(self):
        """Fetch after-metrics, compute delta, then run the evaluator.

        For **VLLM**: Uses Prometheus ``/metrics`` endpoint with before/after
        delta on the ``vllm:spec_decode_*`` counters.

        For **SGLang**: Two strategies are attempted in order:

        1. **Metrics file (preferred)** — if the SGLang server was launched
           with ``--export-metrics-to-file``, read the per-request JSONL log
           and aggregate speculative decoding statistics.
        2. **Prometheus before/after fallback** — fetch
           ``sglang:spec_accept_length``, ``sglang:spec_accept_rate`` (gauges)
           and ``sglang:num_requests_total`` (counter) *before* and *after*
           generation, then back out the benchmark-specific averages from the
           weighted delta.

        The ``eval_specdec`` evaluator receives the pre-computed stats in
        ``eval_config["specdec_stats"]`` and stamps them onto each data point
        in the output JSONL.
        """
        server_address = self._get_server_base_address()
        server_type = self.cfg.server["server_type"]

        specdec_stats: dict[str, Any] | None = None

        if server_type == "sglang":
            # ----- Strategy 1: Metrics file (preferred) -----
            metrics_dir = getattr(self.cfg, "metrics_file_dir", None)
            if metrics_dir:
                metrics_file = find_sglang_metrics_file(metrics_dir)
                if metrics_file:
                    if self._request_ids:
                        request_ids_set = set(self._request_ids)
                        specdec_stats = read_sglang_metrics_file(metrics_file, request_ids_set)
                    else:
                        # No request IDs tracked — read all entries
                        specdec_stats = read_sglang_metrics_file(metrics_file, request_ids=None)
                else:
                    LOG.warning("Could not find SGLang metrics file in %s", metrics_dir)

            # ----- Strategy 2: Prometheus before/after fallback -----
            if specdec_stats is None:
                LOG.info("Falling back to SGLang Prometheus before/after delta…")
                LOG.info("Fetching AFTER SGLang spec-decode metrics from %s/metrics", server_address)
                after_sglang = fetch_sglang_spec_decode_metrics(server_address)

                if self._before_sglang_metrics is not None and after_sglang is not None:
                    specdec_stats = compute_sglang_spec_decode_delta(
                        self._before_sglang_metrics, after_sglang
                    )
                    if specdec_stats is not None:
                        LOG.info(
                            "SGLang Prometheus delta: acceptance_length=%.4f, "
                            "acceptance_rate=%.2f%%, requests=%d, gen_tokens=%d",
                            specdec_stats["acceptance_length"],
                            specdec_stats["acceptance_rate"],
                            specdec_stats["num_drafts"],
                            specdec_stats["draft_tokens"],
                        )
                else:
                    LOG.warning(
                        "Could not compute SGLang Prometheus delta (before=%s, after=%s).",
                        "available" if self._before_sglang_metrics else "missing",
                        "available" if after_sglang else "missing",
                    )
        else:
            # VLLM: Fetch "after" snapshot and compute delta
            LOG.info("Fetching AFTER spec-decode metrics from %s/metrics", server_address)
            after_metrics = fetch_spec_decode_metrics(server_address)

            if self._before_spec_metrics is not None and after_metrics is not None:
                specdec_stats = compute_spec_decode_delta(self._before_spec_metrics, after_metrics)
                if specdec_stats is not None:
                    LOG.info(
                        "Spec-decode delta: drafts=%d, draft_tokens=%d, accepted=%d, "
                        "acceptance_rate=%.2f%%, acceptance_length=%.4f",
                        specdec_stats["num_drafts"],
                        specdec_stats["draft_tokens"],
                        specdec_stats["accepted_tokens"],
                        specdec_stats["acceptance_rate"],
                        specdec_stats["acceptance_length"],
                    )
                    if specdec_stats["per_position_acceptance_rates"]:
                        LOG.info(
                            "Per-position acceptance rates: %s",
                            [f"{r:.4f}" for r in specdec_stats["per_position_acceptance_rates"]],
                        )
            else:
                LOG.warning(
                    "Could not compute spec-decode delta (before=%s, after=%s). "
                    "Speculative decoding may not be enabled on the server.",
                    "available" if self._before_spec_metrics else "missing",
                    "available" if after_metrics else "missing",
                )

        # Inject into eval_config for the evaluator
        self.cfg.eval_config["specdec_stats"] = specdec_stats

        # ----- Copy SGLang metrics file to output directory -----
        if server_type == "sglang" and specdec_stats is not None:
            output_metrics_dir = Path(self.cfg.output_file).parent / "sglang-metrics"
            output_metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir = getattr(self.cfg, "metrics_file_dir", None)
            if metrics_dir:
                metrics_file = find_sglang_metrics_file(metrics_dir)
                if metrics_file:
                    dest = output_metrics_dir / Path(metrics_file).name
                    shutil.copy2(metrics_file, dest)
                    LOG.info("Copied SGLang metrics file to %s", dest)
            # Also write the computed stats as JSON for easy consumption
            stats_file = output_metrics_dir / "specdec_stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(specdec_stats, f, indent=2)
            LOG.info("Wrote specdec stats to %s", stats_file)
            
        super().run_batch_evaluation()


GENERATION_TASK_CLASS = SpecdecGenerationTask


@hydra.main(version_base=None, config_name="base_specdec_generation_config")
def specdec_generation(cfg: SpecdecGenerationConfig):
    cfg = SpecdecGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = SpecdecGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    SpecdecGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        specdec_generation()
