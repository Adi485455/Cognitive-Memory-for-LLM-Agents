from dataclasses import dataclass

from memory_engine.kernel.identity.ids import MemoryId
from memory_engine.kernel.models.lifecycle import LifecycleState
from memory_engine.kernel.models.provenance import Provenance
from memory_engine.kernel.models.temporal import Temporal
from memory_engine.kernel.models.version import Version


@dataclass(frozen=True)
class Memory:
    memory_id: MemoryId
    content: str
    temporal: Temporal
    lifecycle: LifecycleState
    version: Version
    provenance: Provenance
    confidence: float
    importance: float

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Memory content cannot be empty")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance must be between 0.0 and 1.0")