import pytest

from memory_engine.formation.candidates import CandidateFormationService
from memory_engine.formation.episodes import EpisodeManager
from memory_engine.formation.ingestion import IngestionService
from memory_engine.kernel.models.message import Message
from memory_engine.formation.candidates.formation import (
    CandidateFormationService,
)
from memory_engine.kernel.models.memory_type import MemoryType


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


def test_form_candidate_classifies_preference_message(setup):
    formation, message, _ = setup

    # Create a message with preference-related content.
    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python over Java.",
    )
    preference_message = EpisodeManager().add_event(
        event,
        speaker="user",
    )

    candidate = formation.form_candidate(preference_message)

    assert candidate.memory_type == MemoryType.PREFERENCE
    assert candidate.metadata["classification_confidence"] == 0.90
    assert (
        "preference_expression"
        in candidate.metadata["classification_rules"]
    )


def test_form_candidate_keeps_classification_confidence_separate(setup):
    formation, _, _ = setup

    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python over Java.",
    )
    message = EpisodeManager().add_event(event, speaker="user")

    candidate = formation.form_candidate(message)

    assert candidate.metadata["classification_confidence"] == 0.90
    assert candidate.confidence == 0.70
    assert candidate.importance == 0.75


def test_form_candidate_uses_semantic_fallback(setup):
    formation, _, _ = setup

    event = IngestionService().ingest(
        source="chat",
        content="Python supports multiple programming paradigms.",
    )
    message = EpisodeManager().add_event(event, speaker="user")

    candidate = formation.form_candidate(message)

    assert candidate.memory_type == MemoryType.SEMANTIC
    assert candidate.metadata["classification_confidence"] == 0.50
    assert (
        "default_fallback"
        in candidate.metadata["classification_rules"]
    )
def test_form_candidate_assigns_importance_score(setup):
    formation, message, _ = setup

    event = IngestionService().ingest(
        source="chat",
        content="I want to learn machine learning.",
    )
    goal_message = EpisodeManager().add_event(
        event,
        speaker="user",
    )

    candidate = formation.form_candidate(goal_message)

    assert candidate.importance == 0.80
    assert "long_term_goal" in candidate.metadata["importance_rules"]


def test_form_candidate_preserves_default_confidence(setup):
    formation, _, _ = setup

    event = IngestionService().ingest(
        source="chat",
        content="I want to learn machine learning.",
    )
    goal_message = EpisodeManager().add_event(
        event,
        speaker="user",
    )

    candidate = formation.form_candidate(goal_message)

    assert candidate.importance == 0.80
    assert candidate.confidence == 0.5


def test_form_candidate_records_importance_rules(setup):
    formation, _, _ = setup

    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python over Java.",
    )
    preference_message = EpisodeManager().add_event(
        event,
        speaker="user",
    )

    candidate = formation.form_candidate(preference_message)

    assert candidate.importance == 0.75
    assert candidate.metadata["importance_rules"] == (
        "explicit_preference",
    )

def test_form_candidate_assigns_confidence_score(setup):
    formation, _, _ = setup

    event = IngestionService().ingest(
        source="chat",
        content="I completed a machine learning project.",
    )
    message = EpisodeManager().add_event(event, speaker="user")

    candidate = formation.form_candidate(message)

    assert candidate.confidence == 0.75
    assert (
        "explicit_personal_experience"
        in candidate.metadata["confidence_rules"]
    )


def test_form_candidate_records_uncertainty(setup):
    formation, _, _ = setup

    event = IngestionService().ingest(
        source="chat",
        content="I think Python is useful.",
    )
    message = EpisodeManager().add_event(event, speaker="user")

    candidate = formation.form_candidate(message)

    assert candidate.confidence == 0.35
    assert "explicit_uncertainty" in candidate.metadata["confidence_rules"]


def test_form_candidate_keeps_scoring_dimensions_separate(setup):
    formation, _, _ = setup

    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python over Java.",
    )
    message = EpisodeManager().add_event(event, speaker="user")

    candidate = formation.form_candidate(message)

    assert candidate.memory_type == MemoryType.PREFERENCE
    assert candidate.importance == 0.75
    assert candidate.confidence == 0.70
    assert candidate.metadata["classification_confidence"] == 0.90
    assert (
        candidate.metadata["confidence_rules"]
        == ("explicit_preference",)
    )