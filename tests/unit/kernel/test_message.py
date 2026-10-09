import pytest

from memory_engine.kernel.identity.ids import generate_episode_id
from memory_engine.kernel.models.message import Message


def test_message_creation():
    episode_id = generate_episode_id()

    message = Message(
        episode_id=episode_id,
        speaker="user",
        content="I am learning PyTorch.",
        sequence_number=1,
    )

    assert message.episode_id == episode_id
    assert message.speaker == "user"
    assert message.content == "I am learning PyTorch."
    assert message.sequence_number == 1
    assert message.message_id is not None
    assert message.created_at.tzinfo is not None


def test_message_ids_are_unique():
    episode_id = generate_episode_id()

    first = Message(
        episode_id=episode_id,
        speaker="user",
        content="First message",
        sequence_number=1,
    )
    second = Message(
        episode_id=episode_id,
        speaker="assistant",
        content="Second message",
        sequence_number=2,
    )

    assert first.message_id != second.message_id


def test_message_normalizes_speaker():
    message = Message(
        episode_id=generate_episode_id(),
        speaker="  user  ",
        content="Hello",
        sequence_number=1,
    )

    assert message.speaker == "user"


@pytest.mark.parametrize("speaker", ["", "   "])
def test_message_rejects_empty_speaker(speaker):
    with pytest.raises(ValueError):
        Message(
            episode_id=generate_episode_id(),
            speaker=speaker,
            content="Hello",
            sequence_number=1,
        )


@pytest.mark.parametrize("content", ["", "   "])
def test_message_rejects_empty_content(content):
    with pytest.raises(ValueError):
        Message(
            episode_id=generate_episode_id(),
            speaker="user",
            content=content,
            sequence_number=1,
        )


@pytest.mark.parametrize("sequence_number", [0, -1, True, 1.5])
def test_message_rejects_invalid_sequence_number(sequence_number):
    with pytest.raises(ValueError):
        Message(
            episode_id=generate_episode_id(),
            speaker="user",
            content="Hello",
            sequence_number=sequence_number,
        )


def test_message_metadata_cannot_be_modified_directly():
    message = Message(
        episode_id=generate_episode_id(),
        speaker="user",
        content="Hello",
        sequence_number=1,
        metadata={"channel": "chat"},
    )

    with pytest.raises(TypeError):
        message.metadata["channel"] = "api"