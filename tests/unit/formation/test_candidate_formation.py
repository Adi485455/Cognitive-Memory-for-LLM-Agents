import pytest

from memory_engine.formation.candidates import CandidateFormationService
from memory_engine.formation.episodes import EpisodeManager
from memory_engine.formation.ingestion import IngestionService
from memory_engine.kernel.models.message import Message


@pytest.fixture
def setup():
    ingestion = IngestionService()
    episode_manager = EpisodeManager()
    formation = CandidateFormationService()

    event = ingestion.ingest(
        source="chat",
        content="I'm learning PyTorch.",
    )
    message = episode_manager.add_event(event, speaker="user")

    return formation, message, event


def test_form_candidate_from_message(setup):
    formation, message, event = setup

    candidate = formation.form_candidate(message)

    assert candidate.content == message.content
    assert candidate.source_event_ids == (event.event_id,)
    assert candidate.created_at == message.created_at


def test_candidate_preserves_source_metadata(setup):
    formation, message, _ = setup

    candidate = formation.form_candidate(message)

    assert candidate.metadata["source_message_id"] == str(message.message_id)
    assert candidate.metadata["source_episode_id"] == str(message.episode_id)
    assert candidate.metadata["source_speaker"] == message.speaker


def test_forming_twice_creates_distinct_candidates(setup):
    formation, message, _ = setup

    first = formation.form_candidate(message)
    second = formation.form_candidate(message)

    assert first.candidate_id != second.candidate_id


def test_invalid_message_type_is_rejected():
    formation = CandidateFormationService()

    with pytest.raises(TypeError, match="Message"):
        formation.form_candidate("not a message")


def test_missing_event_id_is_rejected(setup):
    formation, message, _ = setup

    invalid_message = Message(
        episode_id=message.episode_id,
        speaker=message.speaker,
        content=message.content,
        sequence_number=message.sequence_number,
        created_at=message.created_at,
        metadata={},
    )

    with pytest.raises(ValueError, match="event_id"):
        formation.form_candidate(invalid_message)


def test_malformed_event_id_is_rejected(setup):
    formation, message, _ = setup

    invalid_message = Message(
        episode_id=message.episode_id,
        speaker=message.speaker,
        content=message.content,
        sequence_number=message.sequence_number,
        created_at=message.created_at,
        metadata={"event_id": "not-a-valid-uuid"},
    )

    with pytest.raises(ValueError, match="invalid event_id"):
        formation.form_candidate(invalid_message)


def test_non_string_event_id_is_rejected(setup):
    formation, message, _ = setup

    invalid_message = Message(
        episode_id=message.episode_id,
        speaker=message.speaker,
        content=message.content,
        sequence_number=message.sequence_number,
        created_at=message.created_at,
        metadata={"event_id": 123},
    )

    with pytest.raises(ValueError, match="string event_id"):
        formation.form_candidate(invalid_message)