from datetime import datetime, timezone

import pytest

from memory_engine.formation.candidates import CandidateMemory
from memory_engine.kernel.identity.ids import generate_event_id
from memory_engine.kernel.models.memory_type import MemoryType


@pytest.fixture
def source_event_id():
    return generate_event_id()


def test_candidate_memory_uses_default_values(source_event_id):
    candidate = CandidateMemory(
        content="Learning PyTorch",
        source_event_ids=(source_event_id,),
    )

    assert candidate.content == "Learning PyTorch"
    assert candidate.normalized_content == "learning pytorch"
    assert candidate.memory_type == MemoryType.SEMANTIC
    assert candidate.importance == 0.5
    assert candidate.confidence == 0.5
    assert candidate.created_at.tzinfo is not None


def test_candidate_ids_are_unique(source_event_id):
    first = CandidateMemory(
        content="Learning PyTorch",
        source_event_ids=(source_event_id,),
    )
    second = CandidateMemory(
        content="Learning PyTorch",
        source_event_ids=(source_event_id,),
    )

    assert first.candidate_id != second.candidate_id


def test_content_is_trimmed(source_event_id):
    candidate = CandidateMemory(
        content="  Learning PyTorch  ",
        source_event_ids=(source_event_id,),
    )

    assert candidate.content == "Learning PyTorch"


def test_normalized_content_uses_casefold_and_whitespace(source_event_id):
    candidate = CandidateMemory(
        content="  LEARNING   PyTorch  ",
        source_event_ids=(source_event_id,),
    )

    assert candidate.normalized_content == "learning pytorch"


def test_custom_normalized_content_is_preserved(source_event_id):
    candidate = CandidateMemory(
        content="Learning PyTorch",
        source_event_ids=(source_event_id,),
        normalized_content="custom representation",
    )

    assert candidate.normalized_content == "custom representation"


def test_empty_content_is_rejected(source_event_id):
    with pytest.raises(ValueError, match="content"):
        CandidateMemory(
            content="   ",
            source_event_ids=(source_event_id,),
        )


def test_empty_source_events_are_rejected():
    with pytest.raises(ValueError, match="at least one source event"):
        CandidateMemory(
            content="Learning PyTorch",
            source_event_ids=(),
        )


def test_source_event_ids_must_be_a_tuple(source_event_id):
    with pytest.raises(TypeError, match="must be a tuple"):
        CandidateMemory(
            content="Learning PyTorch",
            source_event_ids=[source_event_id],
        )


def test_duplicate_source_event_ids_are_rejected(source_event_id):
    with pytest.raises(ValueError, match="unique"):
        CandidateMemory(
            content="Learning PyTorch",
            source_event_ids=(source_event_id, source_event_id),
        )


@pytest.mark.parametrize("field_name", ["importance", "confidence"])
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_scores_outside_range_are_rejected(
    source_event_id, field_name, value
):
    with pytest.raises(ValueError, match=field_name):
        CandidateMemory(
            content="Learning PyTorch",
            source_event_ids=(source_event_id,),
            **{field_name: value},
        )


@pytest.mark.parametrize("field_name", ["importance", "confidence"])
def test_boolean_scores_are_rejected(source_event_id, field_name):
    with pytest.raises(TypeError, match=field_name):
        CandidateMemory(
            content="Learning PyTorch",
            source_event_ids=(source_event_id,),
            **{field_name: True},
        )


def test_naive_created_at_is_rejected(source_event_id):
    with pytest.raises(ValueError, match="timezone-aware"):
        CandidateMemory(
            content="Learning PyTorch",
            source_event_ids=(source_event_id,),
            created_at=datetime(2026, 1, 1),
        )


def test_metadata_snapshot_is_read_only(source_event_id):
    candidate = CandidateMemory(
        content="Learning PyTorch",
        source_event_ids=(source_event_id,),
        metadata={"origin": "chat"},
    )

    with pytest.raises(TypeError):
        candidate.metadata["origin"] = "other"


def test_metadata_input_is_copied(source_event_id):
    metadata = {"origin": "chat"}

    candidate = CandidateMemory(
        content="Learning PyTorch",
        source_event_ids=(source_event_id,),
        metadata=metadata,
    )

    metadata["origin"] = "other"

    assert candidate.metadata["origin"] == "chat"


def test_created_at_can_be_supplied(source_event_id):
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    candidate = CandidateMemory(
        content="Learning PyTorch",
        source_event_ids=(source_event_id,),
        created_at=timestamp,
    )

    assert candidate.created_at == timestamp
def test_invalid_source_event_id_is_rejected():
    with pytest.raises(TypeError, match="valid EventId"):
        CandidateMemory(
            content="Learning PyTorch",
            source_event_ids=("not-a-uuid",),
        )

def test_supported_memory_types():
    from memory_engine.kernel.models.memory_type import MemoryType

    assert MemoryType.EPISODIC.value == "episodic"
    assert MemoryType.SEMANTIC.value == "semantic"
    assert MemoryType.PROCEDURAL.value == "procedural"
    assert MemoryType.PREFERENCE.value == "preference"
    assert MemoryType.TEMPORAL.value == "temporal"
    assert MemoryType.WORKING.value == "working"


def test_invalid_memory_type_is_rejected(source_event_id):
    from memory_engine.kernel.models.memory_type import MemoryType

    with pytest.raises(TypeError, match="MemoryType"):
        CandidateMemory(
            content="Learning PyTorch",
            source_event_ids=(source_event_id,),
            memory_type="random_type",
        )


def test_explicit_memory_type_is_supported(source_event_id):
    from memory_engine.kernel.models.memory_type import MemoryType

    candidate = CandidateMemory(
        content="I prefer practical examples",
        source_event_ids=(source_event_id,),
        memory_type=MemoryType.PREFERENCE,
    )

    assert candidate.memory_type is MemoryType.PREFERENCE