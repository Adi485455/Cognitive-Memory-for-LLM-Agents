from memory_engine.kernel.identity.ids import generate_entity_id
from memory_engine.kernel.models.entity import Entity


def test_entity_can_be_created():
    entity_id = generate_entity_id()

    entity = Entity(
        entity_id=entity_id,
        name="PyTorch",
        entity_type="technology",
    )

    assert entity.entity_id == entity_id
    assert entity.name == "PyTorch"
    assert entity.entity_type == "technology"