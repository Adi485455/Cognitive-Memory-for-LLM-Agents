from datetime import datetime, timezone

import pytest

from memory_engine.formation.ingestion import (
    IngestionService,
    MemoryEvent,
)


@pytest.fixture
def ingestion_service() -> IngestionService:
    return IngestionService()


def test_ingest_creates_memory_event(ingestion_service):
    event = ingestion_service.ingest(
        source="chat",
        content="I am learning PyTorch.",
    )

    assert isinstance(event, MemoryEvent)
    assert event.source == "chat"
    assert event.content == "I am learning PyTorch."
    assert event.event_id is not None
    assert event.received_at.tzinfo is not None


def test_each_ingestion_gets_a_unique_event_id(ingestion_service):
    first = ingestion_service.ingest(
        source="chat",
        content="Learning Python.",
    )
    second = ingestion_service.ingest(
        source="chat",
        content="Learning Python.",
    )

    assert first.event_id != second.event_id


def test_source_is_normalized(ingestion_service):
    event = ingestion_service.ingest(
        source="  chat  ",
        content="Learning Python.",
    )

    assert event.source == "chat"


def test_original_content_is_preserved(ingestion_service):
    content = "  Learning Python.  "

    event = ingestion_service.ingest(
        source="chat",
        content=content,
    )

    assert event.content == content


@pytest.mark.parametrize("source", ["", "   "])
def test_empty_source_is_rejected(ingestion_service, source):
    with pytest.raises(ValueError):
        ingestion_service.ingest(
            source=source,
            content="Learning Python.",
        )


@pytest.mark.parametrize("content", ["", "   "])
def test_empty_content_is_rejected(ingestion_service, content):
    with pytest.raises(ValueError):
        ingestion_service.ingest(
            source="chat",
            content=content,
        )


def test_idempotency_key_is_preserved(ingestion_service):
    event = ingestion_service.ingest(
        source="api",
        content="Learning PyTorch.",
        idempotency_key="request-123",
    )

    assert event.idempotency_key == "request-123"


def test_idempotency_key_is_normalized(ingestion_service):
    event = ingestion_service.ingest(
        source="api",
        content="Learning PyTorch.",
        idempotency_key="  request-123  ",
    )

    assert event.idempotency_key == "request-123"


def test_blank_idempotency_key_is_rejected(ingestion_service):
    with pytest.raises(ValueError):
        ingestion_service.ingest(
            source="api",
            content="Learning PyTorch.",
            idempotency_key="   ",
        )


def test_metadata_is_preserved(ingestion_service):
    metadata = {
        "user_id": "user-1",
        "project": "cognitive-memory",
    }

    event = ingestion_service.ingest(
        source="api",
        content="Working on memory formation.",
        metadata=metadata,
    )

    assert event.metadata["user_id"] == "user-1"
    assert event.metadata["project"] == "cognitive-memory"


def test_event_metadata_cannot_be_modified_directly(ingestion_service):
    event = ingestion_service.ingest(
        source="chat",
        content="Learning Python.",
        metadata={"project": "memory"},
    )

    with pytest.raises(TypeError):
        event.metadata["project"] = "another-project"


def test_event_does_not_share_metadata_mapping(ingestion_service):
    metadata = {"project": "memory"}

    event = ingestion_service.ingest(
        source="chat",
        content="Learning Python.",
        metadata=metadata,
    )

    metadata["project"] = "changed"

    assert event.metadata["project"] == "memory"


def test_custom_received_at_is_preserved(ingestion_service):
    timestamp = datetime(
        2026, 10, 9, 10, 0, tzinfo=timezone.utc
    )

    event = ingestion_service.ingest(
        source="chat",
        content="Learning Python.",
        received_at=timestamp,
    )

    assert event.received_at == timestamp


def test_naive_timestamp_is_rejected(ingestion_service):
    timestamp = datetime(2026, 10, 9, 10, 0)

    with pytest.raises(ValueError):
        ingestion_service.ingest(
            source="chat",
            content="Learning Python.",
            received_at=timestamp,
        )