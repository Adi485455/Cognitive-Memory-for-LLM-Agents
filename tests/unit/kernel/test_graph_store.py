import pytest

from memory_engine.kernel.interfaces.graph_store import GraphStore


def test_graph_store_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        GraphStore()


def test_graph_store_defines_required_operations():
    assert hasattr(GraphStore, "add_relationship")
    assert hasattr(GraphStore, "get_relationship")
    assert hasattr(GraphStore, "delete_relationship")