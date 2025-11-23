from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CandidateScore:
    """Comparable wrapper for Voronoi candidate performance metrics."""

    index: int
    final_score: float
    avg_reduction: float
    worst_ap_improvement: float
    new_ap_client_count: float
    score_std: float
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        assert self.index >= 0, "index must be non-negative"
        assert self.score_std >= 0.0, "score_std must be non-negative"

    @classmethod
    def from_metrics(cls, index: int, metrics: Mapping[str, Any]) -> "CandidateScore":
        """Build a comparable score object from an aggregated metrics mapping."""
        return cls(
            index=int(index),
            final_score=float(metrics.get("final_score", 0.0)),
            avg_reduction=float(metrics.get("avg_reduction_raw_mean", 0.0)),
            worst_ap_improvement=float(metrics.get("worst_ap_improvement_raw_mean", 0.0)),
            new_ap_client_count=float(metrics.get("new_ap_client_count_mean", 0.0)),
            score_std=float(abs(metrics.get("score_std", 0.0))),
            payload=dict(metrics),
        )

    def rank_tuple(self) -> tuple[float, float, float, float, float]:
        """Return the tuple used to rank candidates."""
        return (
            self.final_score,
            self.avg_reduction,
            self.worst_ap_improvement,
            self.new_ap_client_count,
            -self.score_std,
        )


def sort_candidate_scores(scores: Sequence[CandidateScore]) -> list[CandidateScore]:
    """Return scores sorted from best to worst using the required priority order."""
    return sorted(scores, key=lambda score: score.rank_tuple(), reverse=True)


def select_best_candidate(scores: Iterable[CandidateScore]) -> CandidateScore | None:
    """Return the highest-ranked candidate or None when the iterable is empty."""
    best: CandidateScore | None = None
    for score in scores:
        if best is None or score.rank_tuple() > best.rank_tuple():
            best = score
    return best
