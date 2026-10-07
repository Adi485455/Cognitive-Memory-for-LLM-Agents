from dataclasses import dataclass
from datetime import datetime

from memory_engine.kernel.identity.ids import EvidenceId


@dataclass(frozen=True)
class Evidence:
    evidence_id: EvidenceId
    source: str
    content: str
    captured_at: datetime