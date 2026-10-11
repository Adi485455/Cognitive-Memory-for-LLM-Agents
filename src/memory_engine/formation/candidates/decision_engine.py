from memory_engine.formation.candidates.decisions import (
    CandidateDecision,
    CandidateDecisionType,
)
from memory_engine.formation.candidates.matching import MemoryMatch
from memory_engine.formation.candidates.models import CandidateMemory


class CandidateDecisionEngine:
    """Recommends actions for candidate memories."""

    def __init__(
        self,
        minimum_importance: float = 0.30,
        minimum_confidence: float = 0.40,
        strong_match_threshold: float = 0.90,
    ) -> None:
        for name, value in (
            ("minimum_importance", minimum_importance),
            ("minimum_confidence", minimum_confidence),
            ("strong_match_threshold", strong_match_threshold),
        ):
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise TypeError(f"{name} must be numeric")

            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")

        self._minimum_importance = float(minimum_importance)
        self._minimum_confidence = float(minimum_confidence)
        self._strong_match_threshold = float(strong_match_threshold)

    def decide(
        self,
        candidate: CandidateMemory,
        matches: tuple[MemoryMatch, ...] = (),
    ) -> CandidateDecision:
        if not isinstance(candidate, CandidateMemory):
            raise TypeError(
                "candidate must be a CandidateMemory instance"
            )

        if not isinstance(matches, tuple):
            raise TypeError("matches must be a tuple")

        if any(not isinstance(match, MemoryMatch) for match in matches):
            raise TypeError("matches must contain MemoryMatch instances")

        # Low-value candidates can be ignored.
        if candidate.importance < self._minimum_importance:
            return CandidateDecision(
                decision=CandidateDecisionType.IGNORE,
                reason="Candidate importance is below the configured threshold.",
                confidence=0.80,
            )

        # Low-confidence candidates require further review.
        if candidate.confidence < self._minimum_confidence:
            return CandidateDecision(
                decision=CandidateDecisionType.UNCERTAIN,
                reason="Candidate confidence is below the configured threshold.",
                confidence=0.80,
            )

        # Similarity alone cannot establish a semantic duplicate.
        strong_matches = tuple(
            match
            for match in matches
            if match.score >= self._strong_match_threshold
        )

        if strong_matches:
            return CandidateDecision(
                decision=CandidateDecisionType.ESCALATE,
                reason=(
                    "A strong lexical match exists; semantic comparison "
                    "is required before deciding whether to reinforce, "
                    "update, merge, or flag a conflict."
                ),
                target_memory_ids=tuple(
                    match.memory_id for match in strong_matches
                ),
                confidence=0.90,
            )

        # No strong match was found, so propose storing the candidate.
        return CandidateDecision(
            decision=CandidateDecisionType.STORE,
            reason="No strong existing-memory match was found.",
            confidence=0.70,
        )