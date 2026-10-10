from uuid import UUID

from memory_engine.formation.candidates.models import CandidateMemory
from memory_engine.kernel.identity.ids import EventId
from memory_engine.kernel.models.message import Message


class CandidateFormationService:
    """Creates candidate memories from individual messages."""

    def form_candidate(self, message: Message) -> CandidateMemory:
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

        return CandidateMemory(
            content=message.content,
            source_event_ids=(event_id,),
            created_at=message.created_at,
            metadata={
                "source_message_id": str(message.message_id),
                "source_episode_id": str(message.episode_id),
                "source_speaker": message.speaker,
            },
        )