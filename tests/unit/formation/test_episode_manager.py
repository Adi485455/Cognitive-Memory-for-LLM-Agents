import pytest

from memory_engine.formation.episodes import EpisodeManager
from memory_engine.formation.ingestion import IngestionService


@pytest.fixture
def manager():
    return EpisodeManager()


@pytest.fixture
def ingestion():
    return IngestionService()


def create_event(ingestion, content="Learning PyTorch."):
    return ingestion.ingest(
        source="chat",
        content=content,
    )


def test_add_event_creates_episode_and_message(manager, ingestion):
    event = create_event(ingestion)

    message = manager.add_event(
        event,
        speaker="user",
    )

    episode = manager.get_episode(message.episode_id)

    assert episode is not None
    assert episode.source == "chat"
    assert message.content == event.content
    assert message.speaker == "user"
    assert message.sequence_number == 1


def test_multiple_events_can_share_episode(manager, ingestion):
    first_event = create_event(ingestion, "Learning PyTorch.")
    second_event = create_event(ingestion, "Building an image classifier.")

    first_message = manager.add_event(
        first_event,
        speaker="user",
    )

    second_message = manager.add_event(
        second_event,
        speaker="user",
        episode_id=first_message.episode_id,
    )

    assert second_message.episode_id == first_message.episode_id


def test_message_sequence_numbers_increase(manager, ingestion):
    first_event = create_event(ingestion, "First message")
    second_event = create_event(ingestion, "Second message")

    first = manager.add_event(first_event, speaker="user")

    second = manager.add_event(
        second_event,
        speaker="assistant",
        episode_id=first.episode_id,
    )

    third_event = create_event(ingestion, "Third message")

    third = manager.add_event(
        third_event,
        speaker="user",
        episode_id=first.episode_id,
    )

    assert first.sequence_number == 1
    assert second.sequence_number == 2
    assert third.sequence_number == 3


def test_unknown_episode_is_rejected(manager, ingestion):
    from memory_engine.kernel.identity.ids import generate_episode_id

    event = create_event(ingestion)

    with pytest.raises(ValueError, match="Unknown episode ID"):
        manager.add_event(
            event,
            speaker="user",
            episode_id=generate_episode_id(),
        )


def test_same_event_cannot_be_processed_twice(manager, ingestion):
    event = create_event(ingestion)

    manager.add_event(event, speaker="user")

    with pytest.raises(ValueError, match="already been processed"):
        manager.add_event(event, speaker="user")


def test_get_messages_returns_sequence_order(manager, ingestion):
    first_event = create_event(ingestion, "First")
    first_message = manager.add_event(first_event, speaker="user")

    second_event = create_event(ingestion, "Second")
    manager.add_event(
        second_event,
        speaker="assistant",
        episode_id=first_message.episode_id,
    )

    messages = manager.get_messages(first_message.episode_id)

    assert len(messages) == 2
    assert messages[0].content == "First"
    assert messages[1].content == "Second"


def test_get_messages_rejects_unknown_episode(manager):
    from memory_engine.kernel.identity.ids import generate_episode_id

    with pytest.raises(ValueError, match="Unknown episode ID"):
        manager.get_messages(generate_episode_id())


def test_get_message_for_event(manager, ingestion):
    event = create_event(ingestion)

    message = manager.add_event(event, speaker="user")

    assert manager.get_message_for_event(event.event_id) == message


def test_event_source_and_id_are_preserved_in_message_metadata(
    manager,
    ingestion,
):
    event = ingestion.ingest(
        source="api",
        content="Working on the memory system.",
        metadata={"project": "cognitive-memory"},
    )

    message = manager.add_event(event, speaker="user")

    assert message.metadata["source"] == "api"
    assert message.metadata["event_id"] == str(event.event_id)
    assert message.metadata["project"] == "cognitive-memory"