import pytest

from memory_engine.formation.candidates import MemoryTypeClassifier
from memory_engine.kernel.models.memory_type import MemoryType


@pytest.fixture
def classifier():
    return MemoryTypeClassifier()


@pytest.mark.parametrize(
    ("content", "expected_type", "expected_rule"),
    [
        (
            "I prefer practical examples.",
            MemoryType.PREFERENCE,
            "preference_expression",
        ),
        (
            "Run the tests before deploying.",
            MemoryType.PROCEDURAL,
            "procedural_instruction",
        ),
        (
            "My interview is scheduled for tomorrow.",
            MemoryType.TEMPORAL,
            "temporal_expression",
        ),
        (
            "I attended a conference yesterday.",
            MemoryType.EPISODIC,
            "past_experience",
        ),
        (
            "I'm currently debugging the ingestion module.",
            MemoryType.WORKING,
            "current_task_context",
        ),
    ],
)
def test_classifies_supported_patterns(
    classifier, content, expected_type, expected_rule
):
    result = classifier.classify(content)

    assert result.memory_type is expected_type
    assert result.confidence > 0.5
    assert result.matched_rules == (expected_rule,)


def test_unmatched_content_uses_semantic_fallback(classifier):
    result = classifier.classify(
        "Python supports object-oriented programming."
    )

    assert result.memory_type is MemoryType.SEMANTIC
    assert result.confidence == 0.50
    assert result.matched_rules == ("default_fallback",)


def test_preference_rule_takes_priority_over_later_rules(classifier):
    result = classifier.classify(
        "I prefer to run the tests before deploying."
    )

    assert result.memory_type is MemoryType.PREFERENCE


def test_empty_content_is_rejected(classifier):
    with pytest.raises(ValueError, match="cannot be empty"):
        classifier.classify("   ")


def test_non_string_content_is_rejected(classifier):
    with pytest.raises(TypeError, match="must be a string"):
        classifier.classify(None)


def test_classification_result_rejects_invalid_confidence():
    from memory_engine.formation.candidates import ClassificationResult

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        ClassificationResult(
            memory_type=MemoryType.SEMANTIC,
            confidence=1.5,
            matched_rules=("test_rule",),
        )


def test_classification_result_requires_valid_memory_type():
    from memory_engine.formation.candidates import ClassificationResult

    with pytest.raises(TypeError, match="MemoryType"):
        ClassificationResult(
            memory_type="semantic",
            confidence=0.8,
            matched_rules=("test_rule",),
        )