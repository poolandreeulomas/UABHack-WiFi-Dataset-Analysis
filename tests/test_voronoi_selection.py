from __future__ import annotations

import unittest

from dashboard.voronoi_selection import CandidateScore, select_best_candidate, sort_candidate_scores


class CandidateScoreSelectionTest(unittest.TestCase):
    def test_ranking_prefers_highest_final_score(self) -> None:
        better = CandidateScore.from_metrics(
            0,
            {
                "final_score": 0.9,
                "avg_reduction_raw_mean": 0.2,
                "worst_ap_improvement_raw_mean": 0.1,
                "new_ap_client_count_mean": 15,
                "score_std": 0.05,
            },
        )
        worse = CandidateScore.from_metrics(
            1,
            {
                "final_score": 0.7,
                "avg_reduction_raw_mean": 0.4,
                "worst_ap_improvement_raw_mean": 0.2,
                "new_ap_client_count_mean": 30,
                "score_std": 0.02,
            },
        )

        ordered = sort_candidate_scores([worse, better])
        self.assertEqual([score.index for score in ordered], [0, 1])

    def test_ranking_applies_tie_breakers(self) -> None:
        base_metrics = {
            "final_score": 0.75,
            "avg_reduction_raw_mean": 0.15,
            "worst_ap_improvement_raw_mean": 0.1,
            "new_ap_client_count_mean": 10,
            "score_std": 0.04,
        }

        higher_avg = CandidateScore.from_metrics(0, {**base_metrics, "avg_reduction_raw_mean": 0.2})
        higher_worst = CandidateScore.from_metrics(1, {**base_metrics, "worst_ap_improvement_raw_mean": 0.25})
        higher_clients = CandidateScore.from_metrics(2, {**base_metrics, "new_ap_client_count_mean": 40})
        lower_std = CandidateScore.from_metrics(3, {**base_metrics, "score_std": 0.01})

        ordered = sort_candidate_scores([lower_std, higher_clients, higher_worst, higher_avg])
        self.assertEqual([score.index for score in ordered], [0, 1, 2, 3])

    def test_select_best_candidate_handles_empty_iterable(self) -> None:
        self.assertIsNone(select_best_candidate([]))

    def test_select_best_candidate_returns_expected_score(self) -> None:
        scores = [
            CandidateScore.from_metrics(
                idx,
                {
                    "final_score": 0.5 + idx * 0.1,
                    "avg_reduction_raw_mean": 0.1,
                    "worst_ap_improvement_raw_mean": 0.05,
                    "new_ap_client_count_mean": 5 + idx,
                    "score_std": 0.03,
                },
            )
            for idx in range(3)
        ]
        best = select_best_candidate(scores)
        assert best is not None
        self.assertEqual(best.index, 2)


if __name__ == "__main__":
    unittest.main()
