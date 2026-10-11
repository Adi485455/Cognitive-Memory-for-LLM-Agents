import pytest

from memory_engine.formation.candidates import CandidateFormationService
from memory_engine.formation.candidates.validation import (
    CandidateValidator,
)
from memory_engine.formation.episodes import EpisodeManager
from memory_engine.formation.ingestion import IngestionService


@pytest.fixture
def candidate():
    event = IngestionService().ingest(
        source="chat",
        content="I prefer Python over Java.",
    )

    message = EpisodeManager().add_event(
        event,
        speaker="user",
    )

    return CandidateFormationService().form_candidate(message)


@pytest.fixture
def validator():
    return CandidateValidator()


def test_valid_candidate_passes_validation(validator, candidate):
    result = validator.validate(candidate)

    assert result.is_valid is True
    assert result.errors == ()


def test_non_candidate_object_is_rejected(validator):
    with pytest.raises(TypeError, match="CandidateMemory"):
        validator.validate("not a candidate")


def test_candidate_has_required_provenance(candidate, validator):
    result = validator.validate(candidate)

    assert result.is_valid is True

    assert candidate.metadata["source_message_id"]
    assert candidate.metadata["source_episode_id"]
    assert candidate.metadata["source_speaker"]


def test_validation_result_rejects_errors_for_valid_result():
    from memory_engine.formation.candidates.validation import (
        CandidateValidationResult,
    )

    with pytest.raises(ValueError, match="valid result"):
        CandidateValidationResult(
            is_valid=True,
            errors=("missing provenance",),
        )


def test_validation_result_requires_errors_for_invalid_result():
    from memory_engine.formation.candidates.validation import (
        CandidateValidationResult,
    )

    with pytest.raises(ValueError, match="at least one error"):
        CandidateValidationResult(
            is_valid=False,
            errors=(),
        )