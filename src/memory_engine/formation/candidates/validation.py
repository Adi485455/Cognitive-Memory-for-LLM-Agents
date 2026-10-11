from dataclasses import dataclass

from memory_engine.formation.candidates.models import CandidateMemory
from memory_engine.kernel.models.memory_type import MemoryType


@dataclass(frozen=True, slots=True)
class CandidateValidationResult:
    """Result of candidate validation."""

    is_valid: bool
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.is_valid, bool):
            raise TypeError("is_valid must be a boolean")

        if not isinstance(self.errors, tuple):
            raise TypeError("errors must be a tuple")

        if any(
            not isinstance(error, str) or not error.strip()
            for error in self.errors
        ):
            raise ValueError(
                "errors must contain non-empty strings"
            )

        if self.is_valid and self.errors:
            raise ValueError(
                "A valid result cannot contain validation errors"
            )

        if not self.is_valid and not self.errors:
            raise ValueError(
                "An invalid result must contain at least one error"
            )


class CandidateValidator:
    """Validates candidate memory structure and basic eligibility."""

    def validate(self, candidate: CandidateMemory) -> CandidateValidationResult:
        if not isinstance(candidate, CandidateMemory):
            raise TypeError(
                "candidate must be a CandidateMemory instance"
            )

        errors: list[str] = []

        if not candidate.content.strip():
            errors.append("content must not be empty")

        if not candidate.source_event_ids:
            errors.append("at least one source event is required")

        if not isinstance(candidate.memory_type, MemoryType):
            errors.append("memory_type must be a valid MemoryType")

        if not 0.0 <= candidate.importance <= 1.0:
            errors.append(
                "importance must be between 0.0 and 1.0"
            )

        if not 0.0 <= candidate.confidence <= 1.0:
            errors.append(
                "confidence must be between 0.0 and 1.0"
            )

        required_metadata = (
            "source_message_id",
            "source_episode_id",
            "source_speaker",
        )

        for key in required_metadata:
            value = candidate.metadata.get(key)

            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"metadata field '{key}' must be a non-empty string"
                )

        return CandidateValidationResult(
            is_valid=not errors,
            errors=tuple(errors),
        )