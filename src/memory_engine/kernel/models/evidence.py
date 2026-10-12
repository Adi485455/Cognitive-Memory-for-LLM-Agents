from dataclasses import dataclass
from datetime import datetime

from memory_engine.kernel.identity.ids import EvidenceId, EventId


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: EvidenceId
    source: str
    content: str
    captured_at: datetime
    event_id: EventId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Evidence source must be a non-empty string.")

        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Evidence content must be a non-empty string.")

        if not isinstance(self.captured_at, datetime):
            raise TypeError("captured_at must be a datetime.")

        if (
            self.captured_at.tzinfo is None
            or self.captured_at.utcoffset() is None
        ):
            raise ValueError("captured_at must be timezone-aware.")