import pytest

from memory_engine.kernel.models.reinforcement import Reinforcement


def test_reinforcement_defaults():
    reinforcement = Reinforcement()

    assert reinforcement.support_count == 0
    assert reinforcement.confirmation_count == 0
    assert reinforcement.reinforcement_score == 0.0
    assert reinforcement.reinforcement_reason is None


def test_reinforcement_can_store_values():
    reinforcement = Reinforcement(
        support_count=5,
        confirmation_count=3,
        reinforcement_score=0.8,
        reinforcement_reason="Repeated supporting evidence",
    )

    assert reinforcement.support_count == 5
    assert reinforcement.confirmation_count == 3
    assert reinforcement.reinforcement_score == 0.8
    assert reinforcement.reinforcement_reason == "Repeated supporting evidence"


def test_counts_cannot_be_negative():
    with pytest.raises(ValueError):
        Reinforcement(support_count=-1)

    with pytest.raises(ValueError):
        Reinforcement(confirmation_count=-1)


def test_score_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        Reinforcement(reinforcement_score=1.5)

    with pytest.raises(ValueError):
        Reinforcement(reinforcement_score=-0.1)