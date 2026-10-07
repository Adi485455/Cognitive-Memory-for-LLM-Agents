from dataclasses import dataclass

from memory_engine.kernel.identity.ids import EntityId


@dataclass(frozen=True)
class Entity:
    entity_id: EntityId
    name: str
    entity_type: str