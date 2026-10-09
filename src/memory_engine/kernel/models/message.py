from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping

from memory_engine.kernel.identity.ids import (
    EpisodeId,
    MessageId,
    generate_message_id,
)


@dataclass(frozen=True)
class Message:
    """Represents an individual interaction within an episode."""

    episode_id: EpisodeId
    speaker: str
    content: str
    sequence_number: int

    message_id: MessageId = field(default_factory=generate_message_id)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.speaker, str) or not self.speaker.strip():
            raise ValueError("Message speaker cannot be empty")

        object.__setattr__(self, "speaker", self.speaker.strip())

        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Message content cannot be empty")

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 1
        ):
            raise ValueError("sequence_number must be a positive integer")

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