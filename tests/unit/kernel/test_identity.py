from uuid import UUID

from memory_engine.kernel.identity.ids import (
    generate_entity_id,
    generate_memory_id,
)


def test_memory_id_is_uuid():
    memory_id = generate_memory_id()

    assert isinstance(memory_id, UUID)


def test_entity_id_is_uuid():
    entity_id = generate_entity_id()

    assert isinstance(entity_id, UUID)


def test_ids_are_unique():
    first = generate_memory_id()
    second = generate_memory_id()

    assert first != second