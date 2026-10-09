from datetime import datetime
from typing import Any, Mapping

from memory_engine.formation.ingestion.events import MemoryEvent


class IngestionService:
    """
    Receives incoming experiences and converts them into MemoryEvents.

    This service does not create canonical memories, persist events,
    or perform deduplication.
    """

    def ingest(
        self,
        *,
        source: str,
        content: str,
        idempotency_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        received_at: datetime | None = None,
    ) -> MemoryEvent:
        """
        Convert an incoming experience into a validated MemoryEvent.

        Args:
            source: Origin of the experience, such as 'chat' or 'api'.
            content: Original incoming content.
            idempotency_key: Optional caller-provided duplicate-submission key.
            metadata: Optional source-specific metadata.
            received_at: Optional timezone-aware timestamp.

        Returns:
            A validated MemoryEvent.

        Raises:
            ValueError: If required values are invalid.
            TypeError: If a value has an invalid type.
        """
        event_arguments: dict[str, Any] = {
            "source": source,
            "content": content,
            "idempotency_key": idempotency_key,
            "metadata": {} if metadata is None else metadata,
        }

        if received_at is not None:
            event_arguments["received_at"] = received_at

        return MemoryEvent(**event_arguments)