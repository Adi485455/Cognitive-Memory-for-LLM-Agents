from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping

from memory_engine.kernel.identity.ids import EventId, generate_event_id


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """
    Represents an incoming experience before memory formation.

    A MemoryEvent is not a canonical memory. It is an input that
    downstream components may use to form candidate memories.
    """

    source: str
    content: str

    event_id: EventId = field(default_factory=generate_event_id)

    received_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    idempotency_key: str | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate source.
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Event source must be a non-empty string.")

        # Normalize source without changing the original content.
        object.__setattr__(self, "source", self.source.strip())

        # Validate content.
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Event content must be a non-empty string.")

        # Validate timestamp.
        if not isinstance(self.received_at, datetime):
            raise TypeError("received_at must be a datetime.")

        if (
            self.received_at.tzinfo is None
            or self.received_at.utcoffset() is None
        ):
            raise ValueError("received_at must be timezone-aware.")

        # Validate idempotency key.
        if self.idempotency_key is not None:
            if (
                not isinstance(self.idempotency_key, str)
                or not self.idempotency_key.strip()
            ):
                raise ValueError(
                    "idempotency_key must be a non-empty string or None."
                )

            object.__setattr__(
                self,
                "idempotency_key",
                self.idempotency_key.strip(),
            )

        # Copy metadata and prevent direct modification through this object.
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )