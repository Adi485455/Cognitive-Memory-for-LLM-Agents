from uuid import UUID

from memory_engine.formation.candidates.classification import (
    MemoryTypeClassifier,
)
from memory_engine.formation.candidates.confidence import (
    CandidateConfidenceScorer,
)
from memory_engine.kernel.identity.ids import EvidenceId, EventId
from memory_engine.formation.candidates.importance import (
    CandidateImportanceScorer,
)
from memory_engine.formation.candidates.models import CandidateMemory
from memory_engine.kernel.identity.ids import EventId
from memory_engine.kernel.models.message import Message


class CandidateFormationService:
    """Creates candidate memories from individual messages."""

    def __init__(
        self,
        classifier: MemoryTypeClassifier | None = None,
        importance_scorer: CandidateImportanceScorer | None = None,
        confidence_scorer: CandidateConfidenceScorer | None = None,
    ) -> None:
        self._classifier = classifier or MemoryTypeClassifier()
        self._importance_scorer = (
            importance_scorer or CandidateImportanceScorer()
        )
        self._confidence_scorer = (
            confidence_scorer or CandidateConfidenceScorer()
        )

    def form_candidate(self,message: Message,*,source_evidence_ids: tuple[EvidenceId, ...] = (),) -> CandidateMemory:
        if not isinstance(source_evidence_ids, tuple):
            raise TypeError("source_evidence_ids must be a tuple")
        if any(
            not isinstance(evidence_id, UUID)
            for evidence_id in source_evidence_ids
        ):
                raise TypeError(
                    "Each source evidence ID must be a valid EvidenceId"
                )
        if len(set(source_evidence_ids)) != len(source_evidence_ids):
            raise ValueError("Source evidence IDs must be unique")
        if not isinstance(message, Message):
            raise TypeError("message must be a Message instance")

        raw_event_id = message.metadata.get("event_id")

        if not isinstance(raw_event_id, str):
            raise ValueError(
                "Message metadata must contain a string event_id"
            )

        try:
            event_id = EventId(UUID(raw_event_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(
                "Message metadata contains an invalid event_id"
            ) from exc

        classification = self._classifier.classify(message.content)

        importance_result = self._importance_scorer.score(
            message.content
        )

        confidence_result = self._confidence_scorer.score(
            message.content
        )

        return CandidateMemory(
            content=message.content,
            source_event_ids=(event_id,),
            memory_type=classification.memory_type,
            importance=importance_result.score,
            confidence=confidence_result.score,
            source_evidence_ids=source_evidence_ids,
            created_at=message.created_at,
            metadata={
                "source_message_id": str(message.message_id),
                "source_episode_id": str(message.episode_id),
                "source_speaker": message.speaker,
                "classification_confidence": classification.confidence,
                "classification_rules": classification.matched_rules,
                "importance_rules": importance_result.matched_rules,
                "confidence_rules": confidence_result.matched_rules,
            },
        )