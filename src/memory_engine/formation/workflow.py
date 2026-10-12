from dataclasses import dataclass

from memory_engine.formation.candidates import (
    CandidateFormationService,
    CandidateMemory,
)
from memory_engine.formation.candidates.pipeline import (
    CandidateProcessingPipeline,
    CandidateProcessingResult,
)
from memory_engine.formation.episodes import EpisodeManager
from memory_engine.formation.evidence import EvidenceManager
from memory_engine.formation.ingestion import MemoryEvent
from memory_engine.kernel.identity.ids import EpisodeId
from memory_engine.kernel.models.evidence import Evidence
from memory_engine.kernel.models.message import Message
from memory_engine.kernel.models.memory import Memory


@dataclass(frozen=True, slots=True)
class EvidenceAwareCandidateResult:
    """Artifacts produced while processing an incoming event."""

    evidence: Evidence
    message: Message
    candidate: CandidateMemory
    processing: CandidateProcessingResult


class EvidenceAwareCandidateService:
    """Coordinates evidence creation and candidate processing."""

    def __init__(
        self,
        *,
        evidence_manager: EvidenceManager,
        episode_manager: EpisodeManager,
        candidate_formation: CandidateFormationService,
        candidate_pipeline: CandidateProcessingPipeline,
    ) -> None:
        self._evidence_manager = evidence_manager
        self._episode_manager = episode_manager
        self._candidate_formation = candidate_formation
        self._candidate_pipeline = candidate_pipeline

    def process_event(
        self,
        event: MemoryEvent,
        *,
        speaker: str,
        episode_id: EpisodeId | None = None,
        existing_memories: list[Memory] | tuple[Memory, ...] = (),
    ) -> EvidenceAwareCandidateResult:
        """Record evidence, form a candidate, and evaluate it."""

        evidence = self._evidence_manager.record(event)

        message = self._episode_manager.add_event(
            event,
            speaker=speaker,
            episode_id=episode_id,
        )

        candidate = self._candidate_formation.form_candidate(
            message,
            source_evidence_ids=(evidence.evidence_id,),
        )

        processing = self._candidate_pipeline.process(
            candidate,
            existing_memories,
        )

        return EvidenceAwareCandidateResult(
            evidence=evidence,
            message=message,
            candidate=candidate,
            processing=processing,
        )