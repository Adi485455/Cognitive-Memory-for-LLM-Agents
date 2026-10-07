import pytest

from memory_engine.kernel.interfaces.memory_store import MemoryStore


def test_memory_store_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        MemoryStore()


def test_memory_store_defines_required_operations():
    assert hasattr(MemoryStore, "get")
    assert hasattr(MemoryStore, "save")
    assert hasattr(MemoryStore, "delete")