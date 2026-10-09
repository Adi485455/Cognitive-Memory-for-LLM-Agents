from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping

from memory_engine.kernel.identity.ids import EpisodeId, generate_episode_id


@dataclass(frozen=True)
class Episode:
    """Groups related interactions into a single experience."""

    source: str
    episode_id: EpisodeId = field(default_factory=generate_episode_id)
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ended_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Episode source cannot be empty")

        object.__setattr__(self, "source", self.source.strip())

        if not isinstance(self.started_at, datetime):
            raise TypeError("started_at must be a datetime")

        if (
            self.started_at.tzinfo is None
            or self.started_at.utcoffset() is None
        ):
            raise ValueError("started_at must be timezone-aware")

        if self.ended_at is not None:
            if not isinstance(self.ended_at, datetime):
                raise TypeError("ended_at must be a datetime")

            if (
                self.ended_at.tzinfo is None
                or self.ended_at.utcoffset() is None
            ):
                raise ValueError("ended_at must be timezone-aware")

            if self.ended_at < self.started_at:
                raise ValueError("ended_at cannot precede started_at")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )