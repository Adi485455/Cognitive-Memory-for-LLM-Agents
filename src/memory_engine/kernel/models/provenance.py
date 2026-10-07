from dataclasses import dataclass

from memory_engine.kernel.identity.ids import EvidenceId


@dataclass(frozen=True)
class Provenance:
    evidence_ids: tuple[EvidenceId, ...]