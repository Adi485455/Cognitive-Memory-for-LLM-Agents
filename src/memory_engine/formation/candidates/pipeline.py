from dataclasses import dataclass

from memory_engine.formation.candidates.decision_engine import (
    CandidateDecisionEngine,
)
from memory_engine.formation.candidates.decisions import (
    CandidateDecision,
    CandidateDecisionType,
)
from memory_engine.formation.candidates.matching import (
    CandidateMemoryMatcher,
    MemoryMatch,
)
from memory_engine.formation.candidates.models import CandidateMemory
from memory_engine.formation.candidates.validation import (
    CandidateValidationResult,
    CandidateValidator,
)
from memory_engine.kernel.models.memory import Memory


@dataclass(frozen=True, slots=True)
class CandidateProcessingResult:
    """Result of processing a candidate through the pipeline."""

    candidate: CandidateMemory
    validation: CandidateValidationResult
    matches: tuple[MemoryMatch, ...]
    decision: CandidateDecision


class CandidateProcessingPipeline:
    """Coordinates validation, matching, and candidate decisions."""

    def __init__(
        self,
        validator: CandidateValidator | None = None,
        matcher: CandidateMemoryMatcher | None = None,
        decision_engine: CandidateDecisionEngine | None = None,
    ) -> None:
        self._validator = validator or CandidateValidator()
        self._matcher = matcher or CandidateMemoryMatcher()
        self._decision_engine = (
            decision_engine or CandidateDecisionEngine()
        )

    def process(
        self,
        candidate: CandidateMemory,
        existing_memories: list[Memory] | tuple[Memory, ...] = (),
    ) -> CandidateProcessingResult:
        validation = self._validator.validate(candidate)

        if not validation.is_valid:
            decision = CandidateDecision(
                decision=CandidateDecisionType.IGNORE,
                reason=(
                    "Candidate validation failed: "
                    + "; ".join(validation.errors)
                ),
                confidence=1.0,
            )

            return CandidateProcessingResult(
                candidate=candidate,
                validation=validation,
                matches=(),
                decision=decision,
            )

        matches = self._matcher.find_matches(
            candidate,
            existing_memories,
        )

        decision = self._decision_engine.decide(
            candidate,
            matches,
        )

        return CandidateProcessingResult(
            candidate=candidate,
            validation=validation,
            matches=matches,
            decision=decision,
        )