import pytest

from memory_engine.formation.candidates.confidence import (
    CandidateConfidenceScorer,
)


@pytest.fixture
def scorer():
    return CandidateConfidenceScorer()


def test_explicit_uncertainty_receives_lower_score(scorer):
    result = scorer.score("I think Python is useful.")

    assert result.score == 0.35
    assert "explicit_uncertainty" in result.matched_rules


def test_personal_experience_receives_higher_score(scorer):
    result = scorer.score("I completed a machine learning project.")

    assert result.score == 0.75
    assert "explicit_personal_experience" in result.matched_rules


def test_preference_receives_heuristic_score(scorer):
    result = scorer.score("I prefer Python over Java.")

    assert result.score == 0.70
    assert "explicit_preference" in result.matched_rules


def test_uncertainty_takes_precedence(scorer):
    result = scorer.score(
        "I think I completed a machine learning project."
    )

    assert result.score == 0.35
    assert "explicit_uncertainty" in result.matched_rules
    assert "explicit_personal_experience" in result.matched_rules


def test_unmatched_content_uses_fallback(scorer):
    result = scorer.score(
        "Python supports multiple programming paradigms."
    )

    assert result.score == 0.50
    assert result.matched_rules == ("default_fallback",)


def test_non_string_content_is_rejected(scorer):
    with pytest.raises(TypeError, match="content"):
        scorer.score(123)


def test_empty_content_is_rejected(scorer):
    with pytest.raises(ValueError, match="empty"):
        scorer.score("   ")
        