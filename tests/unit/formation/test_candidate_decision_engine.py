import pytest

from memory_engine.formation.candidates import CandidateFormationService
from memory_engine.formation.candidates.decisions import (
    CandidateDecision,
    CandidateDecisionType,
)
from memory_engine.formation.candidates.decision_engine import (
    CandidateDecisionEngine,
)
from memory_engine.formation.candidates.matching import MemoryMatch
from memory_engine.formation.episodes import EpisodeManager
from memory_engine.formation.ingestion import IngestionService
from memory_engine.kernel.identity.ids import generate_memory_id


@pytest.fixture
def engine():
    return CandidateDecisionEngine()


@pytest.fixture
def candidate():
    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python over Java.",
    )
    message = EpisodeManager().add_event(event, speaker="user")

    return CandidateFormationService().form_candidate(message)


def test_candidate_without_matches_proposes_store(engine, candidate):
    decision = engine.decide(candidate)

    assert decision.decision == CandidateDecisionType.STORE
    assert decision.target_memory_ids == ()


def test_strong_match_requires_escalation(engine, candidate):
    memory_id = generate_memory_id()

    match = MemoryMatch(
        memory_id=memory_id,
        score=0.95,
        match_type="normalized_exact",
    )

    decision = engine.decide(candidate, (match,))

    assert decision.decision == CandidateDecisionType.ESCALATE
    assert decision.target_memory_ids == (memory_id,)


def test_weak_match_does_not_trigger_escalation(engine, candidate):
    match = MemoryMatch(
        memory_id=generate_memory_id(),
        score=0.50,
        match_type="token_jaccard",
    )

    decision = engine.decide(candidate, (match,))

    assert decision.decision == CandidateDecisionType.STORE


def test_invalid_candidate_is_rejected(engine):
    with pytest.raises(TypeError, match="CandidateMemory"):
        engine.decide("not a candidate")


def test_invalid_matches_collection_is_rejected(engine, candidate):
    with pytest.raises(TypeError, match="tuple"):
        engine.decide(candidate, [])


def test_invalid_decision_confidence_is_rejected():
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        CandidateDecision(
            decision=CandidateDecisionType.STORE,
            reason="Test",
            confidence=1.5,
        )