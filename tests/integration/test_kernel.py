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
from memory_engine.kernel.validation.validators import validate_memory


def test_kernel_can_build_and_validate_memory():
    now = datetime.now(timezone.utc)

    evidence_id = generate_evidence_id()
    memory_id = generate_memory_id()

    provenance = Provenance(
        evidence_ids=(evidence_id,),
    )

    temporal = Temporal(
        created_at=now,
        learned_at=now,
        event_time=now,
        valid_from=now,
    )

    memory = Memory(
        memory_id=memory_id,
        content="I use Python for machine learning.",
        temporal=temporal,
        lifecycle=LifecycleState.ACTIVE,
        version=Version(number=1),
        provenance=provenance,
        confidence=0.9,
        importance=0.8,
    )

    validate_memory(memory)

    assert memory.memory_id == memory_id
    assert memory.provenance.evidence_ids == (evidence_id,)
    assert memory.version.number == 1
    assert memory.lifecycle == LifecycleState.ACTIVE