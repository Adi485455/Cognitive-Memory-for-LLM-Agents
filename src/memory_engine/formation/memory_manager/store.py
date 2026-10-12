from threading import RLock

from memory_engine.kernel.identity.ids import MemoryId
from memory_engine.kernel.interfaces.memory_store import MemoryStore
from memory_engine.kernel.models.memory import Memory


class InMemoryMemoryStore(MemoryStore):
    """Thread-safe, in-memory storage for canonical memories."""

    def __init__(self) -> None:
        self._memories: dict[MemoryId, Memory] = {}
        self._lock = RLock()

    def get(self, memory_id: MemoryId) -> Memory | None:
        with self._lock:
            return self._memories.get(memory_id)

    def save(self, memory: Memory) -> None:
        if not isinstance(memory, Memory):
            raise TypeError("memory must be a Memory instance")

        with self._lock:
            existing = self._memories.get(memory.memory_id)

            if existing is not None and existing != memory:
                raise ValueError(
                    "Cannot overwrite a different memory using the same ID"
                )

            self._memories[memory.memory_id] = memory

    def delete(self, memory_id: MemoryId) -> None:
        with self._lock:
            self._memories.pop(memory_id, None)