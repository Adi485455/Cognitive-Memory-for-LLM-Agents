import pytest

from memory_engine.formation.candidates import CandidateFormationService
from memory_engine.formation.candidates.decision_engine import (
    CandidateDecisionEngine,
)
from memory_engine.formation.candidates.decisions import (
    CandidateDecisionType,
)
from memory_engine.formation.candidates.pipeline import (
    CandidateProcessingPipeline,
)
from memory_engine.formation.episodes import EpisodeManager
from memory_engine.formation.evidence import (
    EvidenceManager,
    InMemoryEvidenceStore,
)
from memory_engine.formation.ingestion import IngestionService
from memory_engine.formation.memory_manager import (
    InMemoryMemoryStore,
    MemoryManager,
)
from memory_engine.formation.workflow import EvidenceAwareCandidateService


@pytest.fixture
def workflow():
    return EvidenceAwareCandidateService(
        evidence_manager=EvidenceManager(InMemoryEvidenceStore()),
        episode_manager=EpisodeManager(),
        candidate_formation=CandidateFormationService(),
        candidate_pipeline=CandidateProcessingPipeline(),
        memory_manager=MemoryManager(InMemoryMemoryStore()),
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
    service = IngestionService()

    first_event = service.ingest(
        source="chat",
        content="I use Python.",
    )

    first_result = workflow.process_event(
        first_event,
        speaker="user",
    )

    second_event = service.ingest(
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


def test_store_decision_creates_canonical_memory(workflow):
    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python over Java.",
    )

    result = workflow.process_event(event, speaker="user")

    assert result.processing.decision.decision is CandidateDecisionType.STORE
    assert result.canonical_memory is not None
    assert result.canonical_memory.content == event.content
    assert result.canonical_memory.provenance.evidence_ids == (
        result.evidence.evidence_id,
    )


def test_escalation_does_not_create_canonical_memory():
    pipeline = CandidateProcessingPipeline(
        decision_engine=CandidateDecisionEngine(
            strong_match_threshold=0.1,
        ),
    )

    workflow = EvidenceAwareCandidateService(
        evidence_manager=EvidenceManager(InMemoryEvidenceStore()),
        episode_manager=EpisodeManager(),
        candidate_formation=CandidateFormationService(),
        candidate_pipeline=pipeline,
        memory_manager=MemoryManager(InMemoryMemoryStore()),
    )

    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python over Java.",
    )

    # A matching memory causes escalation instead of storage.
    from datetime import datetime, timezone

    from memory_engine.kernel.identity.ids import (
        generate_evidence_id,
        generate_memory_id,
    )
    from memory_engine.kernel.models.lifecycle import LifecycleState
    from memory_engine.kernel.models.memory import Memory
    from memory_engine.kernel.models.provenance import Provenance
    from memory_engine.kernel.models.temporal import Temporal
    from memory_engine.kernel.models.version import Version

    now = datetime.now(timezone.utc)

    existing = Memory(
        memory_id=generate_memory_id(),
        content="I prefer Python over Java.",
        temporal=Temporal(created_at=now, learned_at=now),
        lifecycle=LifecycleState.ACTIVE,
        version=Version(number=1),
        provenance=Provenance(
            evidence_ids=(generate_evidence_id(),),
        ),
        confidence=0.8,
        importance=0.8,
    )

    result = workflow.process_event(
        event,
        speaker="user",
        existing_memories=[existing],
    )

    assert result.processing.decision.decision is CandidateDecisionType.ESCALATE
    assert result.canonical_memory is None