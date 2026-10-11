import pytest

from memory_engine.formation.candidates import CandidateFormationService
from memory_engine.formation.candidates.decisions import (
    CandidateDecisionType,
)
from memory_engine.formation.candidates.matching import (
    CandidateMemoryMatcher,
)
from memory_engine.formation.candidates.pipeline import (
    CandidateProcessingPipeline,
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
def candidate():
    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python over Java.",
    )
    message = EpisodeManager().add_event(event, speaker="user")

    return CandidateFormationService().form_candidate(message)


@pytest.fixture
def pipeline():
    return CandidateProcessingPipeline()


def make_memory(content: str) -> Memory:
    from datetime import datetime, timezone

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


def test_pipeline_processes_valid_candidate(pipeline, candidate):
    result = pipeline.process(candidate)

    assert result.validation.is_valid
    assert result.matches == ()
    assert result.decision.decision == CandidateDecisionType.STORE


def test_pipeline_finds_existing_memory_match(pipeline, candidate):
    memory = make_memory("I prefer Python over Java")

    result = pipeline.process(candidate, [memory])

    assert result.validation.is_valid
    assert len(result.matches) == 1
    assert result.decision.decision == CandidateDecisionType.ESCALATE


def test_pipeline_accepts_empty_existing_memory_collection(
    pipeline,
    candidate,
):
    result = pipeline.process(candidate, ())

    assert result.matches == ()
    assert result.decision.decision == CandidateDecisionType.STORE


def test_pipeline_returns_structured_result(pipeline, candidate):
    result = pipeline.process(candidate)

    assert result.candidate is candidate
    assert result.validation is not None
    assert isinstance(result.matches, tuple)
    assert result.decision is not None