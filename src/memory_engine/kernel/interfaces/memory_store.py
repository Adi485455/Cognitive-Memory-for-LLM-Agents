from abc import ABC, abstractmethod

from memory_engine.kernel.identity.ids import EvidenceId, MemoryId
from memory_engine.kernel.models.memory import Memory


class MemoryStore(ABC):

    @abstractmethod
    def get(self, memory_id: MemoryId) -> Memory | None:
        pass

    @abstractmethod
    def save(self, memory: Memory) -> None:
        pass

    @abstractmethod
    def delete(self, memory_id: MemoryId) -> None:
        pass

    @abstractmethod
    def find_by_evidence_ids(
        self,
        evidence_ids: tuple[EvidenceId, ...],
    ) -> tuple[Memory, ...]:
        pass