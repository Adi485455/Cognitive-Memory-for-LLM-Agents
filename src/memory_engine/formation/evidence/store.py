from threading import RLock

from memory_engine.kernel.identity.ids import EvidenceId
from memory_engine.kernel.interfaces.evidence_store import EvidenceStore
from memory_engine.kernel.models.evidence import Evidence


class InMemoryEvidenceStore(EvidenceStore):
    """Thread-safe, in-memory evidence store for development and testing."""

    def __init__(self) -> None:
        self._evidence: dict[EvidenceId, Evidence] = {}
        self._lock = RLock()

    def get(self, evidence_id: EvidenceId) -> Evidence | None:
        with self._lock:
            return self._evidence.get(evidence_id)

    def save(self, evidence: Evidence) -> None:
        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must be an Evidence instance.")

        with self._lock:
            existing = self._evidence.get(evidence.evidence_id)

            if existing is not None and existing != evidence:
                raise ValueError(
                    "Cannot overwrite different evidence using the same ID."
                )

            self._evidence[evidence.evidence_id] = evidence

    def delete(self, evidence_id: EvidenceId) -> None:
        with self._lock:
            self._evidence.pop(evidence_id, None)