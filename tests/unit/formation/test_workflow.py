import pytest

from memory_engine.formation.candidates import CandidateFormationService
from memory_engine.formation.candidates.pipeline import (
    CandidateProcessingPipeline,
)
from memory_engine.formation.episodes import EpisodeManager
from memory_engine.formation.evidence import (
    EvidenceManager,
    InMemoryEvidenceStore,
)
from memory_engine.formation.ingestion import IngestionService
from memory_engine.formation.workflow import EvidenceAwareCandidateService


@pytest.fixture
def workflow():
    evidence_manager = EvidenceManager(InMemoryEvidenceStore())
    episode_manager = EpisodeManager()

    return EvidenceAwareCandidateService(
        evidence_manager=evidence_manager,
        episode_manager=episode_manager,
        candidate_formation=CandidateFormationService(),
        candidate_pipeline=CandidateProcessingPipeline(),
    )


def test_workflow_preserves_evidence_traceability(workflow):
    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python over Java.",
    )

    result = workflow.process_event(event, speaker="user")

    assert result.evidence.event_id == event.event_id
    assert result.candidate.source_event_ids == (event.event_id,)
    assert result.candidate.source_evidence_ids == (
        result.evidence.evidence_id,
    )


def test_workflow_returns_all_processing_artifacts(workflow):
    event = IngestionService().ingest(
        source="chat",
        content="I use Python for machine learning.",
    )

    result = workflow.process_event(event, speaker="user")

    assert result.message.content == event.content
    assert result.candidate.content == event.content
    assert result.processing.candidate is result.candidate
    assert result.processing.validation.is_valid


def test_workflow_can_process_event_in_existing_episode(workflow):
    event_service = IngestionService()

    first_event = event_service.ingest(
        source="chat",
        content="I use Python.",
    )

    first_result = workflow.process_event(
        first_event,
        speaker="user",
    )

    second_event = event_service.ingest(
        source="chat",
        content="I use machine learning.",
    )

    second_result = workflow.process_event(
        second_event,
        speaker="user",
        episode_id=first_result.message.episode_id,
    )

    assert (
        second_result.message.episode_id
        == first_result.message.episode_id
    )
    assert second_result.message.sequence_number == 2


def test_workflow_rejects_processing_same_event_twice(workflow):
    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python.",
    )

    workflow.process_event(event, speaker="user")

    with pytest.raises(ValueError, match="already been processed"):
        workflow.process_event(event, speaker="user")