from datetime import datetime, timezone

import pytest

from memory_engine.formation.evidence import (
    EvidenceManager,
    InMemoryEvidenceStore,
)
from memory_engine.formation.ingestion import IngestionService


@pytest.fixture
def store():
    return InMemoryEvidenceStore()


@pytest.fixture
def manager(store):
    return EvidenceManager(store)


def test_record_creates_evidence_from_event(manager):
    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python over Java.",
    )

    evidence = manager.record(event)

    assert evidence.event_id == event.event_id
    assert evidence.source == event.source
    assert evidence.content == event.content
    assert evidence.captured_at == event.received_at


def test_record_saves_evidence_to_store(store, manager):
    event = IngestionService().ingest(
        source="chat",
        content="I use Python for machine learning.",
    )

    evidence = manager.record(event)

    assert store.get(evidence.evidence_id) == evidence


def test_recording_same_event_returns_same_evidence(manager):
    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python.",
    )

    first = manager.record(event)
    second = manager.record(event)

    assert first.evidence_id == second.evidence_id
    assert first == second


def test_reusing_event_id_with_different_content_is_rejected(manager):
    service = IngestionService()

    original = service.ingest(
        source="chat",
        content="I prefer Python.",
    )
    manager.record(original)

    changed_event = type(original)(
        event_id=original.event_id,
        source="chat",
        content="I prefer Java.",
        received_at=original.received_at,
    )

    with pytest.raises(ValueError, match="reused"):
        manager.record(changed_event)


def test_evidence_store_rejects_non_evidence(store):
    with pytest.raises(TypeError):
        store.save("not evidence")


def test_evidence_requires_timezone_aware_timestamp():
    from memory_engine.kernel.identity.ids import generate_evidence_id
    from memory_engine.kernel.models.evidence import Evidence

    with pytest.raises(ValueError, match="timezone-aware"):
        Evidence(
            evidence_id=generate_evidence_id(),
            source="chat",
            content="Some evidence.",
            captured_at=datetime(2026, 1, 1),
        )