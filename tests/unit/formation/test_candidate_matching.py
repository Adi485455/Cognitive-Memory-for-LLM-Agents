from datetime import datetime, timezone

import pytest

from memory_engine.formation.candidates import CandidateFormationService
from memory_engine.formation.candidates.matching import (
    CandidateMemoryMatcher,
)
from memory_engine.formation.episodes import EpisodeManager
from memory_engine.formation.ingestion import IngestionService
from memory_engine.kernel.identity.ids import (
    generate_evidence_id,
    generate_memory_id,
)
from memory_engine.kernel.models.lifecycle import LifecycleState
from memory_engine.kernel.models.memory import Memory
from memory_engine.kernel.models.provenance import Provenance
from memory_engine.kernel.models.temporal import Temporal
from memory_engine.kernel.models.version import Version


@pytest.fixture
def matcher():
    return CandidateMemoryMatcher()


@pytest.fixture
def candidate():
    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python over Java.",
    )
    message = EpisodeManager().add_event(event, speaker="user")

    return CandidateFormationService().form_candidate(message)


def make_memory(content: str) -> Memory:
    now = datetime.now(timezone.utc)

    return Memory(
        memory_id=generate_memory_id(),
        content=content,
        temporal=Temporal(
            created_at=now,
            learned_at=now,
        ),
        lifecycle=LifecycleState.ACTIVE,
        version=Version(number=1),
        provenance=Provenance(
            evidence_ids=(generate_evidence_id(),),
        ),
        confidence=0.8,
        importance=0.7,
    )


def test_normalized_exact_match(matcher, candidate):
    memory = make_memory("I prefer Python over Java")

    matches = matcher.find_matches(candidate, [memory])

    assert len(matches) == 1
    assert matches[0].memory_id == memory.memory_id
    assert matches[0].score == 1.0
    assert matches[0].match_type == "normalized_exact"


def test_case_insensitive_match(matcher, candidate):
    memory = make_memory("I PREFER PYTHON OVER JAVA.")

    matches = matcher.find_matches(candidate, [memory])

    assert len(matches) == 1
    assert matches[0].score == 1.0


def test_unrelated_memory_is_not_matched(matcher, candidate):
    memory = make_memory("My favorite food is pizza.")

    matches = matcher.find_matches(candidate, [memory])

    assert matches == ()


def test_results_are_sorted_by_score(matcher, candidate):
    exact_memory = make_memory("I prefer Python over Java")
    partial_memory = make_memory("I prefer Python")
    unrelated_memory = make_memory("I enjoy playing cricket")

    matches = matcher.find_matches(
        candidate,
        [partial_memory, unrelated_memory, exact_memory],
    )

    assert len(matches) == 2
    assert matches[0].memory_id == exact_memory.memory_id
    assert matches[0].score >= matches[1].score


def test_match_threshold_can_be_configured(candidate):
    matcher = CandidateMemoryMatcher(min_score=0.9)
    memory = make_memory("I prefer Python")

    matches = matcher.find_matches(candidate, [memory])

    assert matches == ()


def test_invalid_candidate_is_rejected(matcher):
    with pytest.raises(TypeError, match="CandidateMemory"):
        matcher.find_matches("not a candidate", [])


def test_invalid_memory_collection_is_rejected(matcher, candidate):
    with pytest.raises(TypeError, match="list or tuple"):
        matcher.find_matches(candidate, "not a collection")


def test_invalid_memory_entry_is_rejected(matcher, candidate):
    with pytest.raises(TypeError, match="every existing item"):
        matcher.find_matches(candidate, ["not a memory"])


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        CandidateMemoryMatcher(min_score=1.5)