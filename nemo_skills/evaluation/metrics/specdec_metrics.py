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

import logging
from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class SpecdecMetrics(BaseMetrics):
    """Metrics for SPEED-Bench speculative decoding evaluation.

    Reads per-data-point speculative decoding statistics that were computed
    by the generation task using a before/after delta of the server's
    Prometheus ``/metrics`` counters and stamped by the ``eval_specdec``
    evaluator.

    Key metrics (computed from Prometheus counter deltas):

    * **acceptance_length** (AL) — ``1 + delta_accepted / delta_drafts``.
      The "+1" accounts for the target model's verified token that is always
      emitted even on a full rejection.
    * **acceptance_rate** — ``(delta_accepted / delta_draft_tokens) * 100``.
    * **per_position_acceptance_rates** — per-position acceptance probability,
      computed as ``delta_per_pos[i] / delta_drafts``.

    The class also tracks per-category breakdowns using the ``category``
    field present in SPEED-Bench data points.
    """

    def __init__(self):
        super().__init__(compute_no_answer=False)

    def reset(self):
        super().reset()
        # Aggregate speculative decoding metrics (from before/after delta)
        self.acceptance_length: float | None = None
        self.acceptance_rate: float | None = None
        self.num_drafts: int | None = None
        self.draft_tokens: int | None = None
        self.accepted_tokens: int | None = None
        self.per_position_acceptance_rates: list[float] | None = None

        # Per-category tracking
        self.category_counts: dict[str, int] = defaultdict(int)
        self.category_total_gen_tokens: dict[str, int] = defaultdict(int)
        self.category_total_gen_time: dict[str, float] = defaultdict(float)

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Return a dummy score dict — speculative decoding has no correctness metric."""
        return {}

    def update(self, predictions: list[dict]) -> None:
        """Update the evaluation results with the current element.

        Args:
            predictions: Aggregated predictions across all generations.
                Each prediction should contain speculative decoding metrics
                stamped by the ``eval_specdec`` evaluator.
        """
        super().update(predictions)

        # Capture aggregate server metrics from the first prediction
        # (all data points carry the same server-level values from the delta)
        pred = predictions[0]
        if self.acceptance_length is None and pred.get("acceptance_length") is not None:
            self.acceptance_length = pred["acceptance_length"]
            self.acceptance_rate = pred.get("acceptance_rate")
            self.num_drafts = pred.get("num_drafts")
            self.draft_tokens = pred.get("draft_tokens")
            self.accepted_tokens = pred.get("accepted_tokens")
            self.per_position_acceptance_rates = pred.get("per_position_acceptance_rates")

        # Track per-category statistics
        category = pred.get("category", "unknown")
        self.category_counts[category] += 1
        for p in predictions:
            gen_tokens = p.get("num_generated_tokens", 0)
            gen_time = p.get("generation_time", 0.0)
            self.category_total_gen_tokens[category] += gen_tokens
            self.category_total_gen_time[category] += gen_time

    def get_metrics(self) -> dict:
        """Get all computed metrics including speculative decoding statistics.

        Returns:
            Nested dict of evaluation mode → metric name → value.
        """
        metrics_dict = {}

        # We use a single evaluation mode for specdec metrics
        agg_mode = "specdec"
        agg_dict: dict = {}
        self.update_common_metrics(agg_dict)

        # Server-level speculative decoding metrics (already computed as deltas)
        if self.acceptance_length is not None:
            agg_dict["acceptance_length"] = self.acceptance_length
        if self.acceptance_rate is not None:
            agg_dict["acceptance_rate"] = self.acceptance_rate  # already a percentage
        if self.num_drafts is not None:
            agg_dict["num_drafts"] = self.num_drafts
        if self.draft_tokens is not None:
            agg_dict["draft_tokens"] = self.draft_tokens
        if self.accepted_tokens is not None:
            agg_dict["accepted_tokens"] = self.accepted_tokens
        if self.per_position_acceptance_rates:
            agg_dict["per_position_acceptance_rates"] = self.per_position_acceptance_rates

        metrics_dict[agg_mode] = agg_dict

        # Per-category breakdown
        if self.category_counts:
            category_results: dict = {}
            for category, count in sorted(self.category_counts.items()):
                cat_metrics: dict = {"num_entries": count}
                total_tokens = self.category_total_gen_tokens.get(category, 0)
                total_time = self.category_total_gen_time.get(category, 0.0)
                if count > 0:
                    cat_metrics["avg_gen_tokens"] = total_tokens / count
                if total_time > 0:
                    cat_metrics["tokens_per_second"] = total_tokens / total_time
                category_results[category] = cat_metrics

            agg_dict["category_breakdown"] = category_results

        return metrics_dict

    def evaluations_to_print(self) -> list[str]:
        """Return which evaluation modes should be printed."""
        return ["specdec"]

    def metrics_to_print(self) -> dict:
        """Control which metrics are displayed in the summary table."""
        return {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "acceptance_length": as_float,
            "acceptance_rate": as_float,
        }
