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

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int, as_percentage
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
        # Per-category tracking
        self.category_counts = defaultdict(int)

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {
            "spec_draft_tokens": prediction.get("draft_tokens", 0),
            "spec_accepted_tokens": prediction.get("accepted_tokens", 0),
            "spec_num_drafts": prediction.get("num_drafts", 0),
            "spec_acceptance_length": prediction.get("acceptance_length", 0.0),
            "spec_acceptance_rate": prediction.get("acceptance_rate", 0.0),
        }
    
    # def _compute_pass_at_k(
    #     self, predictions: list[dict], predicted_answers: list[str] | None = None, eval_dict: dict | None = None
    # ):
    #     """
    #     Get pass@k metrics for speculative decoding.

    #     Args:
    #         predictions (list): List of generated predictions.
    #             Will call `_get_score_dict` to get specdec metrics per prediction.
    #         predicted_answers (Optional[list]): List of the answers that will be used to compute no_answer metric.
    #         eval_dict (Optional[dict]): Dictionary to store aggregated metrics.
    #             By default will use self.eval_dict.
    #     """
    #     if eval_dict is None:
    #         eval_dict = self.eval_dict
    #     score_dicts = [self._get_score_dict(pred) for pred in predictions]

    #     for score_method in score_dicts[0].keys():
    #         scores_list = [correctness_dict[score_method] for correctness_dict in score_dicts]
    #         self.all_scores[score_method].append(scores_list)

    #         instance_pass_scores = []
    #         instance_pass_avg_scores = []

    #         for k in range(1, len(predictions) + 1):
    #             instance_pass_score = max(scores_list[:k])
    #             instance_pass_scores.append(instance_pass_score)
    #             instance_pass_avg_scores.append(sum(scores_list[:k]) / k)
        
    #         eval_dict[f"pass@{k}"][score_method] = sum(instance_pass_scores) / len(instance_pass_scores)
    #         eval_dict[f"pass@1[avg-of-{k}]"][score_method] = sum(instance_pass_avg_scores) / len(instance_pass_avg_scores)
    #         eval_dict[f"pass@{k}[{predictions[0].get("category", "unknown")}]"][score_method] = sum(instance_pass_scores) / len(instance_pass_scores)
    #         eval_dict[f"pass@1[avg-of-{k}][{predictions[0].get("category", "unknown")}]"] = sum(instance_pass_avg_scores) / len(instance_pass_avg_scores)

    def update(self, predictions: list[dict]) -> None:
        """Update the evaluation results with the current element.

        Args:
            predictions: Aggregated predictions across all generations.
                Each prediction should contain speculative decoding metrics
                stamped by the ``eval_specdec`` evaluator.
        """
        super().update(predictions)
        self._compute_pass_at_k(
            predictions=predictions, 
            predicted_answers=[pred.get("generation", None) for pred in predictions]
        )

        # Track per-category statistics
        category = predictions[0].get("category", "unknown")
        self.category_counts[category] += 1

    def get_metrics(self) -> dict:
        """Get all computed metrics including speculative decoding statistics.

        Returns:
            Nested dict of evaluation mode → metric name → value.
        """
        metrics_dict = {}

        for agg_mode, agg_metric_dict in self.eval_dict.items():
            metrics_dict[agg_mode] = {}
            self.update_common_metrics(metrics_dict[agg_mode])
            for metric_key, metric_value in agg_metric_dict.items():
                if metric_key.startswith("spec_"):
                    metrics_dict[agg_mode][metric_key] = metric_value / self.total
                else:
                    metrics_dict[agg_mode][metric_key] = metric_value
        self._add_std_metrics(metrics_dict)

        return metrics_dict

    def metrics_to_print(self) -> dict:
        """Control which metrics are displayed in the summary table."""
        metrics_to_print = {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "spec_acceptance_length": as_float,
            "spec_acceptance_rate": as_float,
        }
        if self.compute_no_answer:
            metrics_to_print["no_answer"] = as_percentage
        return metrics_to_print
