from memory_engine.kernel.models.lifecycle import LifecycleState


def test_all_lifecycle_states_exist():
    assert LifecycleState.CANDIDATE.value == "candidate"
    assert LifecycleState.ACTIVE.value == "active"
    assert LifecycleState.SUPERSEDED.value == "superseded"
    assert LifecycleState.ARCHIVED.value == "archived"
    assert LifecycleState.REJECTED.value == "rejected"
    assert LifecycleState.DELETED.value == "deleted"


def test_lifecycle_state_is_string_compatible():
    assert LifecycleState.ACTIVE == "active"


def test_superseded_is_different_from_deleted():
    assert LifecycleState.SUPERSEDED != LifecycleState.DELETED