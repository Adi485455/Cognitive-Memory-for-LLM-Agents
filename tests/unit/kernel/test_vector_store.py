import pytest

from memory_engine.kernel.interfaces.vector_store import VectorStore


def test_vector_store_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        VectorStore()


def test_vector_store_defines_required_operations():
    assert hasattr(VectorStore, "upsert")
    assert hasattr(VectorStore, "search")
    assert hasattr(VectorStore, "delete")