from datetime import datetime, timezone

import pytest

from memory_engine.kernel.identity.ids import (
    generate_evidence_id,
    generate_memory_id,
)
from memory_engine.kernel.models.lifecycle import LifecycleState
from memory_engine.kernel.models.memory import Memory
from memory_engine.kernel.models.provenance import Provenance
from memory_engine.kernel.models.temporal import Temporal
from memory_engine.kernel.models.version import Version


def create_test_memory() -> Memory:
    now = datetime.now(timezone.utc)

    temporal = Temporal(
        created_at=now,
        learned_at=now,
    )

    provenance = Provenance(
        evidence_ids=(generate_evidence_id(),),
    )

    return Memory(
        memory_id=generate_memory_id(),
        content="I use Python for machine learning.",
        temporal=temporal,
        lifecycle=LifecycleState.ACTIVE,
        version=Version(number=1),
        provenance=provenance,
        confidence=0.9,
        importance=0.8,
    )


def test_memory_can_be_created():
    memory = create_test_memory()

    assert memory.content == "I use Python for machine learning."
    assert memory.lifecycle == LifecycleState.ACTIVE
    assert memory.version.number == 1
    assert memory.confidence == 0.9
    assert memory.importance == 0.8


def test_memory_requires_non_empty_content():
    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError):
        Memory(
            memory_id=generate_memory_id(),
            content="   ",
            temporal=Temporal(
                created_at=now,
                learned_at=now,
            ),
            lifecycle=LifecycleState.ACTIVE,
            version=Version(number=1),
            provenance=Provenance(
                evidence_ids=(generate_evidence_id(),),
            ),
            confidence=0.9,
            importance=0.8,
        )


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        memory = create_test_memory()
        Memory(
            memory_id=memory.memory_id,
            content=memory.content,
            temporal=memory.temporal,
            lifecycle=memory.lifecycle,
            version=memory.version,
            provenance=memory.provenance,
            confidence=1.5,
            importance=memory.importance,
        )


def test_importance_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        memory = create_test_memory()
        Memory(
            memory_id=memory.memory_id,
            content=memory.content,
            temporal=memory.temporal,
            lifecycle=memory.lifecycle,
            version=memory.version,
            provenance=memory.provenance,
            confidence=memory.confidence,
            importance=-0.1,
        )


def test_memory_identity_is_stable_across_versions():
    memory = create_test_memory()

    updated_memory = Memory(
        memory_id=memory.memory_id,
        content="I use Python 3.14 for machine learning.",
        temporal=memory.temporal,
        lifecycle=memory.lifecycle,
        version=Version(number=2),
        provenance=memory.provenance,
        confidence=0.95,
        importance=memory.importance,
    )

    assert updated_memory.memory_id == memory.memory_id
    assert updated_memory.version.number == 2
    assert updated_memory.content != memory.content