from datetime import datetime, timezone
from uuid import uuid4

import pytest

from memory_engine.formation.candidates.decisions import (
    CandidateDecisionType,
)
from memory_engine.formation.candidates.models import CandidateMemory
from memory_engine.formation.memory_manager import (
    InMemoryMemoryStore,
    MemoryManager,
)
from memory_engine.kernel.identity.ids import (
    EvidenceId,
    EventId,
)


@pytest.fixture
def store():
    return InMemoryMemoryStore()


@pytest.fixture
def manager(store):
    return MemoryManager(store)


@pytest.fixture
def candidate():
    return CandidateMemory(
        content="I prefer Python over Java.",
        source_event_ids=(EventId(uuid4()),),
        source_evidence_ids=(EvidenceId(uuid4()),),
        confidence=0.7,
        importance=0.75,
        metadata={
            "source_message_id": str(uuid4()),
            "source_episode_id": str(uuid4()),
            "source_speaker": "user",
        },
    )


def test_store_candidate_creates_active_memory(manager, store, candidate):
    memory = manager.store_candidate(
        candidate,
        decision=CandidateDecisionType.STORE,
    )

    assert memory.content == candidate.content
    assert memory.lifecycle.value == "active"
    assert memory.version.number == 1
    assert memory.confidence == candidate.confidence
    assert memory.importance == candidate.importance
    assert store.get(memory.memory_id) == memory


def test_store_candidate_preserves_evidence_provenance(
    manager,
    candidate,
):
    memory = manager.store_candidate(
        candidate,
        decision=CandidateDecisionType.STORE,
    )

    assert memory.provenance.evidence_ids == candidate.source_evidence_ids


def test_store_candidate_uses_supplied_timestamp(manager, candidate):
    timestamp = datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc)

    memory = manager.store_candidate(
        candidate,
        decision=CandidateDecisionType.STORE,
        now=timestamp,
    )

    assert memory.temporal.created_at == timestamp
    assert memory.temporal.learned_at == timestamp


@pytest.mark.parametrize(
    "decision",
    [
        CandidateDecisionType.IGNORE,
        CandidateDecisionType.UPDATE,
        CandidateDecisionType.MERGE,
        CandidateDecisionType.SUPERSEDE,
        CandidateDecisionType.CONFLICT,
        CandidateDecisionType.CREATE_RELATION,
        CandidateDecisionType.UNCERTAIN,
        CandidateDecisionType.ESCALATE,
    ],
)
def test_non_store_decisions_are_rejected(manager, candidate, decision):
    with pytest.raises(ValueError, match="STORE"):
        manager.store_candidate(candidate, decision=decision)


def test_candidate_without_evidence_is_rejected(manager):
    candidate = CandidateMemory(
        content="A fact without evidence",
        source_event_ids=(EventId(uuid4()),),
        metadata={
            "source_message_id": str(uuid4()),
            "source_episode_id": str(uuid4()),
            "source_speaker": "user",
        },
    )

    with pytest.raises(ValueError, match="evidence provenance"):
        manager.store_candidate(
            candidate,
            decision=CandidateDecisionType.STORE,
        )


def test_naive_timestamp_is_rejected(manager, candidate):
    with pytest.raises(ValueError, match="timezone-aware"):
        manager.store_candidate(
            candidate,
            decision=CandidateDecisionType.STORE,
            now=datetime(2026, 10, 12, 10, 0),
        )