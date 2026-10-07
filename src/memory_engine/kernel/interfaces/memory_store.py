from abc import ABC, abstractmethod

from memory_engine.kernel.identity.ids import MemoryId
from memory_engine.kernel.models.memory import Memory


class MemoryStore(ABC):

    @abstractmethod
    def get(self, memory_id: MemoryId) -> Memory | None:
        """Retrieve a memory by its stable ID."""
        raise NotImplementedError

    @abstractmethod
    def save(self, memory: Memory) -> None:
        """Persist a memory."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, memory_id: MemoryId) -> None:
        """Delete a memory from the store."""
        raise NotImplementedError