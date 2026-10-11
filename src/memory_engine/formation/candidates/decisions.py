from dataclasses import dataclass
from enum import Enum

from memory_engine.kernel.identity.ids import MemoryId


class CandidateDecisionType(str, Enum):
    STORE = "store"
    IGNORE = "ignore"
    UPDATE = "update"
    MERGE = "merge"
    SUPERSEDE = "supersede"
    CONFLICT = "conflict"
    CREATE_RELATION = "create_relation"
    UNCERTAIN = "uncertain"
    ESCALATE = "escalate"


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    """A proposed decision for a candidate memory."""

    decision: CandidateDecisionType
    reason: str
    target_memory_ids: tuple[MemoryId, ...] = ()
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.decision, CandidateDecisionType):
            raise TypeError(
                "decision must be a CandidateDecisionType"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

        if not isinstance(self.target_memory_ids, tuple):
            raise TypeError("target_memory_ids must be a tuple")

        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise TypeError("confidence must be numeric")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0"
            )

        object.__setattr__(self, "confidence", float(self.confidence))