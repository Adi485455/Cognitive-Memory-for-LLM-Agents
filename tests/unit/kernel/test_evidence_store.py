import pytest

from memory_engine.kernel.interfaces.evidence_store import EvidenceStore


def test_evidence_store_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        EvidenceStore()


def test_evidence_store_defines_required_operations():
    assert hasattr(EvidenceStore, "get")
    assert hasattr(EvidenceStore, "save")
    assert hasattr(EvidenceStore, "delete")