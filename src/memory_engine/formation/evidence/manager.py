from memory_engine.formation.ingestion.events import MemoryEvent
from memory_engine.kernel.identity.ids import EvidenceId, generate_evidence_id
from memory_engine.kernel.interfaces.evidence_store import EvidenceStore
from memory_engine.kernel.models.evidence import Evidence


class EvidenceManager:
    """Creates evidence records from ingestion events."""

    def __init__(self, store: EvidenceStore) -> None:
        if not isinstance(store, EvidenceStore):
            raise TypeError("store must implement EvidenceStore.")

        self._store = store
        self._event_to_evidence: dict[object, EvidenceId] = {}

    def record(self, event: MemoryEvent) -> Evidence:
        if not isinstance(event, MemoryEvent):
            raise TypeError("event must be a MemoryEvent instance.")

        existing_evidence_id = self._event_to_evidence.get(event.event_id)

        if existing_evidence_id is not None:
            existing_evidence = self._store.get(existing_evidence_id)

            if existing_evidence is not None:
                if (
                    existing_evidence.source != event.source
                    or existing_evidence.content != event.content
                    or existing_evidence.captured_at != event.received_at
                ):
                    raise ValueError(
                        "Event ID was reused with different event content."
                    )

                return existing_evidence

        evidence = Evidence(
            evidence_id=generate_evidence_id(),
            event_id=event.event_id,
            source=event.source,
            content=event.content,
            captured_at=event.received_at,
        )

        self._store.save(evidence)
        self._event_to_evidence[event.event_id] = evidence.evidence_id

        return evidence