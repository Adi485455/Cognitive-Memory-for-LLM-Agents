from memory_engine.kernel.identity.ids import generate_evidence_id
from memory_engine.kernel.models.provenance import Provenance


def test_provenance_can_reference_evidence():
    evidence_id = generate_evidence_id()

    provenance = Provenance(
        evidence_ids=(evidence_id,),
    )

    assert provenance.evidence_ids == (evidence_id,)
    assert evidence_id in provenance.evidence_ids


def test_provenance_can_reference_multiple_evidence_items():
    first = generate_evidence_id()
    second = generate_evidence_id()

    provenance = Provenance(
        evidence_ids=(first, second),
    )

    assert len(provenance.evidence_ids) == 2
    assert first in provenance.evidence_ids
    assert second in provenance.evidence_ids