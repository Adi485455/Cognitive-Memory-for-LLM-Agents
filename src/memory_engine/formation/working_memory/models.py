from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from memory_engine.kernel.identity.ids import MessageId
from memory_engine.kernel.models.message import Message


class WorkingMemoryStatus(str, Enum):
    ACTIVE = "active"
    PROMOTED = "promoted"
    DEMOTED = "demoted"


@dataclass(frozen=True)
class WorkingMemoryEntry:
    """A message tracked within the temporary working set."""

    message: Message
    priority: float = 0.5
    status: WorkingMemoryStatus = WorkingMemoryStatus.ACTIVE
    inserted_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def message_id(self) -> MessageId:
        return self.message.message_id

    def __post_init__(self) -> None:
        if not isinstance(self.message, Message):
            raise TypeError("message must be a Message instance")

        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("priority must be between 0.0 and 1.0")

        if not isinstance(self.status, WorkingMemoryStatus):
            raise TypeError("status must be a WorkingMemoryStatus")

        if (
            self.inserted_at.tzinfo is None
            or self.inserted_at.utcoffset() is None
        ):
            raise ValueError("inserted_at must be timezone-aware")