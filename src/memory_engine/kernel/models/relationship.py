from dataclasses import dataclass
from datetime import datetime

from memory_engine.kernel.identity.ids import (
    EntityId,
    MemoryId,
    RelationshipId,
)


@dataclass(frozen=True)
class Relationship:
    relationship_id: RelationshipId
    source_id: EntityId | MemoryId
    target_id: EntityId | MemoryId
    relationship_type: str
    confidence: float
    inference_type: str
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        if self.valid_from is not None and self.valid_until is not None:
            if self.valid_until < self.valid_from:
                raise ValueError(
                    "valid_until cannot be earlier than valid_from"
                )