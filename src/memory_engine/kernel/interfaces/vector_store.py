from abc import ABC, abstractmethod

from memory_engine.kernel.identity.ids import MemoryId


class VectorStore(ABC):

    @abstractmethod
    def upsert(
        self,
        memory_id: MemoryId,
        embedding: list[float],
    ) -> None:
        """Insert or update a memory embedding."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        embedding: list[float],
        limit: int = 10,
    ) -> list[MemoryId]:
        """Find memory IDs similar to an embedding."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, memory_id: MemoryId) -> None:
        """Remove a memory embedding."""
        raise NotImplementedError