from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID

from memory_engine.kernel.identity.ids import (
    CandidateId,
    EventId,
    generate_candidate_id,
)


@dataclass(frozen=True, slots=True)
class CandidateMemory:
    content: str
    source_event_ids: tuple[EventId, ...]

    candidate_id: CandidateId = field(
        default_factory=generate_candidate_id
    )
    normalized_content: str = ""
    memory_type: str = "semantic"
    importance: float = 0.5
    confidence: float = 0.5
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Candidate content cannot be empty")

        object.__setattr__(self, "content", self.content.strip())

        if not isinstance(self.source_event_ids, tuple):
            raise TypeError("source_event_ids must be a tuple")

        if any(
            not isinstance(event_id, UUID)
            for event_id in self.source_event_ids):
            raise TypeError("Each source event ID must be a valid EventId")
        
        

        if not self.source_event_ids:
            raise ValueError(
                "A candidate must reference at least one source event"
            )

        if len(set(self.source_event_ids)) != len(self.source_event_ids):
            raise ValueError("Source event IDs must be unique")

        if not isinstance(self.memory_type, str) or not self.memory_type.strip():
            raise ValueError("memory_type cannot be empty")

        object.__setattr__(self, "memory_type", self.memory_type.strip())

        normalized = self.normalized_content.strip()

        if not normalized:
            normalized = " ".join(self.content.casefold().split())

        object.__setattr__(self, "normalized_content", normalized)

        for field_name in ("importance", "confidence"):
            value = getattr(self, field_name)

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a number")

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0.0 and 1.0"
                )

            object.__setattr__(self, field_name, float(value))

        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime")

        if (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError("created_at must be timezone-aware")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )