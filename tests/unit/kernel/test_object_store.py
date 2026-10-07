import pytest

from memory_engine.kernel.interfaces.object_store import ObjectStore


def test_object_store_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ObjectStore()


def test_object_store_defines_required_operations():
    assert hasattr(ObjectStore, "put")
    assert hasattr(ObjectStore, "get")
    assert hasattr(ObjectStore, "delete")