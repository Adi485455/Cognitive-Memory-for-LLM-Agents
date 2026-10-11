import pytest

from memory_engine.formation.candidates.importance import (
    CandidateImportanceScorer,
)


@pytest.fixture
def scorer():
    return CandidateImportanceScorer()


def test_explicit_preference_receives_high_importance(scorer):
    result = scorer.score("I prefer Python over Java.")

    assert result.score == 0.75
    assert "explicit_preference" in result.matched_rules


def test_long_term_goal_receives_high_importance(scorer):
    result = scorer.score("I want to learn machine learning.")

    assert result.score == 0.80
    assert "long_term_goal" in result.matched_rules


def test_ongoing_project_receives_high_importance(scorer):
    result = scorer.score(
        "I am currently building an LLM memory system."
    )

    assert result.score == 0.75
    assert "ongoing_project" in result.matched_rules


def test_temporary_event_receives_low_importance(scorer):
    result = scorer.score("I had lunch today.")

    assert result.score == 0.20
    assert "temporary_event" in result.matched_rules


def test_unmatched_content_uses_default_score(scorer):
    result = scorer.score(
        "Python supports multiple programming paradigms."
    )

    assert result.score == 0.50
    assert result.matched_rules == ("default_fallback",)


def test_multiple_matching_rules_are_preserved(scorer):
    result = scorer.score(
        "I prefer Python, and my goal is to become an ML engineer."
    )

    assert result.score == 0.80
    assert "explicit_preference" in result.matched_rules
    assert "long_term_goal" in result.matched_rules


def test_non_string_content_is_rejected(scorer):
    with pytest.raises(TypeError, match="content"):
        scorer.score(123)


def test_empty_content_is_rejected(scorer):
    with pytest.raises(ValueError, match="empty"):
        scorer.score("   ")