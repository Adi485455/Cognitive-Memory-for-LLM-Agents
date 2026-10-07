from datetime import datetime, timezone

from memory_engine.kernel.identity.ids import (
    generate_evidence_id,
    generate_memory_id,
)
from memory_engine.kernel.models.lifecycle import LifecycleState
from memory_engine.kernel.models.memory import Memory
from memory_engine.kernel.models.provenance import Provenance
from memory_engine.kernel.models.temporal import Temporal
from memory_engine.kernel.models.version import Version
from memory_engine.kernel.validation.invariants import (
    validate_memory_invariants,
)


def create_valid_memory() -> Memory:
    now = datetime.now(timezone.utc)

    return Memory(
        memory_id=generate_memory_id(),
        content="I use Python for machine learning.",
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


def test_valid_memory_passes_invariants():
    memory = create_valid_memory()

    validate_memory_invariants(memory)


def test_memory_requires_provenance():
    memory = create_valid_memory()

    invalid_memory = Memory(
        memory_id=memory.memory_id,
        content=memory.content,
        temporal=memory.temporal,
        lifecycle=memory.lifecycle,
        version=memory.version,
        provenance=Provenance(evidence_ids=()),
        confidence=memory.confidence,
        importance=memory.importance,
    )

    try:
        validate_memory_invariants(invalid_memory)
        assert False, "Expected ValueError"
    except ValueError as error:
        assert "provenance" in str(error)


def test_memory_with_valid_temporal_range_passes():
    now = datetime.now(timezone.utc)

    memory = create_valid_memory()

    temporal = Temporal(
        created_at=now,
        learned_at=now,
        valid_from=now,
    )

    valid_memory = Memory(
        memory_id=memory.memory_id,
        content=memory.content,
        temporal=temporal,
        lifecycle=memory.lifecycle,
        version=memory.version,
        provenance=memory.provenance,
        confidence=memory.confidence,
        importance=memory.importance,
    )

    validate_memory_invariants(valid_memory)