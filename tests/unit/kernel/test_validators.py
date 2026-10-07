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
from memory_engine.kernel.validation.validators import validate_memory


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


def test_validate_memory_accepts_valid_memory():
    memory = create_valid_memory()

    validate_memory(memory)


def test_validate_memory_rejects_missing_provenance():
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

    with pytest.raises(ValueError):
        validate_memory(invalid_memory)