from threading import RLock

from memory_engine.formation.ingestion.events import MemoryEvent
from memory_engine.kernel.identity.ids import EpisodeId, EventId
from memory_engine.kernel.models.episode import Episode
from memory_engine.kernel.models.message import Message


class EpisodeManager:
    """
    Coordinates episodes and messages in memory.

    This implementation is process-local and non-durable.
    Persistent storage and cross-process coordination will be
    introduced through storage contracts later.
    """

    def __init__(self) -> None:
        self._episodes: dict[EpisodeId, Episode] = {}
        self._messages: dict[EpisodeId, list[Message]] = {}
        self._event_to_message: dict[EventId, Message] = {}
        self._lock = RLock()

    def create_episode(self, source: str) -> Episode:
        """Create and register a new episode."""
        episode = Episode(source=source)

        with self._lock:
            self._episodes[episode.episode_id] = episode
            self._messages[episode.episode_id] = []

        return episode

    def add_event(
        self,
        event: MemoryEvent,
        *,
        speaker: str,
        episode_id: EpisodeId | None = None,
    ) -> Message:
        """
        Add an event as a message to an episode.

        If episode_id is omitted, a new episode is created.

        An event may be processed only once by this manager.
        """
        with self._lock:
            if event.event_id in self._event_to_message:
                raise ValueError(
                    f"Event {event.event_id} has already been processed"
                )

            if episode_id is None:
                episode = Episode(source=event.source)
                self._episodes[episode.episode_id] = episode
                self._messages[episode.episode_id] = []
            else:
                episode = self._episodes.get(episode_id)

                if episode is None:
                    raise ValueError(
                        f"Unknown episode ID: {episode_id}"
                    )

            messages = self._messages[episode.episode_id]

            message = Message(
                episode_id=episode.episode_id,
                speaker=speaker,
                content=event.content,
                sequence_number=len(messages) + 1,
                created_at=event.received_at,
                metadata={
                    **event.metadata,
                    "event_id": str(event.event_id),
                    "source": event.source,
                },
            )

            messages.append(message)
            self._event_to_message[event.event_id] = message

            return message

    def get_episode(self, episode_id: EpisodeId) -> Episode | None:
        """Return an episode, or None if it is unknown."""
        with self._lock:
            return self._episodes.get(episode_id)

    def get_messages(self, episode_id: EpisodeId) -> tuple[Message, ...]:
        """Return messages in their assigned sequence order."""
        with self._lock:
            if episode_id not in self._episodes:
                raise ValueError(f"Unknown episode ID: {episode_id}")

            return tuple(self._messages[episode_id])

    def get_message_for_event(self, event_id: EventId) -> Message | None:
        """Find the message created from an event."""
        with self._lock:
            return self._event_to_message.get(event_id)