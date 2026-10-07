from datetime import datetime, timezone

from memory_engine.kernel.identity.ids import generate_evidence_id
from memory_engine.kernel.models.evidence import Evidence


def test_evidence_can_be_created():
    evidence_id = generate_evidence_id()
    captured_at = datetime.now(timezone.utc)

    evidence = Evidence(
        evidence_id=evidence_id,
        source="conversation",
        content="I use Python for machine learning.",
        captured_at=captured_at,
    )

    assert evidence.evidence_id == evidence_id
    assert evidence.source == "conversation"
    assert evidence.content == "I use Python for machine learning."
    assert evidence.captured_at == captured_at