from abc import ABC, abstractmethod

from memory_engine.kernel.identity.ids import EvidenceId
from memory_engine.kernel.models.evidence import Evidence


class EvidenceStore(ABC):

    @abstractmethod
    def get(self, evidence_id: EvidenceId) -> Evidence | None:
        """Retrieve evidence by its stable ID."""
        raise NotImplementedError

    @abstractmethod
    def save(self, evidence: Evidence) -> None:
        """Persist evidence."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, evidence_id: EvidenceId) -> None:
        """Delete evidence from the store."""
        raise NotImplementedError