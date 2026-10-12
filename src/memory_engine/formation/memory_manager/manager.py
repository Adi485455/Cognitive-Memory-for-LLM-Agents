from datetime import datetime, timezone

from memory_engine.formation.candidates.decisions import (
    CandidateDecisionType,
)
from memory_engine.formation.candidates.models import CandidateMemory
from memory_engine.kernel.identity.ids import (
    MemoryId,
    generate_memory_id,
)
from memory_engine.kernel.interfaces.memory_store import MemoryStore
from memory_engine.kernel.models.lifecycle import LifecycleState
from memory_engine.kernel.models.memory import Memory
from memory_engine.kernel.models.provenance import Provenance
from memory_engine.kernel.models.temporal import Temporal
from memory_engine.kernel.models.version import Version


class MemoryManager:
    """Manages the creation of canonical memories."""

    def __init__(self, store: MemoryStore) -> None:
        if not isinstance(store, MemoryStore):
            raise TypeError("store must implement MemoryStore")

        self._store = store

    def store_candidate(
        self,
        candidate: CandidateMemory,
        *,
        decision: CandidateDecisionType,
        now: datetime | None = None,
    ) -> Memory:
        """Create a canonical memory from an approved candidate."""

        if not isinstance(candidate, CandidateMemory):
            raise TypeError(
                "candidate must be a CandidateMemory instance"
            )

        if decision is not CandidateDecisionType.STORE:
            raise ValueError(
                "Only candidates with a STORE decision can be stored"
            )

        if not candidate.source_evidence_ids:
            raise ValueError(
                "Cannot store a candidate without evidence provenance"
            )

        timestamp = now or datetime.now(timezone.utc)

        if not isinstance(timestamp, datetime):
            raise TypeError("now must be a datetime")

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("now must be timezone-aware")

        memory = Memory(
            memory_id=generate_memory_id(),
            content=candidate.content,
            temporal=Temporal(
                created_at=timestamp,
                learned_at=timestamp,
            ),
            lifecycle=LifecycleState.ACTIVE,
            version=Version(number=1),
            provenance=Provenance(
                evidence_ids=candidate.source_evidence_ids,
            ),
            confidence=candidate.confidence,
            importance=candidate.importance,
        )

        self._store.save(memory)

        return memory