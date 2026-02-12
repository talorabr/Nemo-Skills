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
import logging
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import hydra
import requests

from nemo_skills.inference.generate import (
    GenerationTask,
    GenerationTaskConfig,
    InferenceConfig,
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
                                accepted_per_pos[pos] = accepted_per_pos.get(pos, 0) + val
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
            before_val = before.accepted_per_pos.get(pos, 0)
            after_val = after.accepted_per_pos.get(pos, before_val)
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

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_specdec_generation_config", node=SpecdecGenerationConfig)


class SpecdecGenerationTask(GenerationTask):
    """Custom generation task for SPEED-Bench speculative decoding evaluation.

    Captures speculative decoding counters from the server's ``/metrics``
    Prometheus endpoint *before* and *after* generation, then computes the
    delta to derive accurate acceptance length (AL), acceptance rate, and
    per-position conditional acceptance rates.
    """

    def __init__(self, cfg: SpecdecGenerationConfig):
        super().__init__(cfg)
        self._before_spec_metrics: SpecDecodeMetrics | None = None

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
        turns = data_point.get("turns", [])
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
        """
        turns = data_point.get("turns", [])
        is_multiturn = data_point.get("multiturn", False) and len(turns) > 1

        if not is_multiturn:
            return await super().process_single_datapoint(data_point, all_data)

        # --- Multi-turn generation ---
        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            inference_params = dict(self.cfg.inference)

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
            generation_text = result.get("generation", "")
            num_tokens = result.get("num_generated_tokens", 0)

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
        server_cfg = dict(self.cfg.server)
        base_url = server_cfg.get("base_url")
        if base_url:
            # Strip the /v1 suffix if present to get the root address
            return base_url.rstrip("/").removesuffix("/v1")
        host = server_cfg.get("host", "127.0.0.1")
        port = server_cfg.get("port", "5000")
        return f"http://{host}:{port}"

    def wait_for_server(self):
        """Wait for the server, then snapshot speculative decoding counters.

        This captures the "before" metrics so that after generation we can
        compute a clean delta isolated to our generation run.
        """
        super().wait_for_server()
        base_url = self._get_server_base_address()
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

        The ``eval_specdec`` evaluator receives the pre-computed delta stats
        in ``eval_config["specdec_stats"]`` and stamps them onto each data
        point in the output JSONL.
        """
        server_address = self._get_server_base_address()
        server_type = dict(self.cfg.server).get("server_type", "vllm")

        # Fetch "after" snapshot and compute delta
        LOG.info("Fetching AFTER spec-decode metrics from %s/metrics", server_address)
        after_metrics = fetch_spec_decode_metrics(server_address)

        specdec_stats: dict[str, Any] | None = None
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
        self.cfg.eval_config["server_address"] = server_address
        self.cfg.eval_config["server_type"] = server_type
        self.cfg.eval_config["specdec_stats"] = specdec_stats

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
