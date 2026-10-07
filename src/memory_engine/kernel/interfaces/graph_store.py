from abc import ABC, abstractmethod

from memory_engine.kernel.identity.ids import RelationshipId


class GraphStore(ABC):

    @abstractmethod
    def add_relationship(
        self,
        relationship_id: RelationshipId,
    ) -> None:
        """Add a relationship to the graph representation."""
        raise NotImplementedError

    @abstractmethod
    def get_relationship(
        self,
        relationship_id: RelationshipId,
    ) -> RelationshipId | None:
        """Retrieve a relationship by its stable ID."""
        raise NotImplementedError

    @abstractmethod
    def delete_relationship(
        self,
        relationship_id: RelationshipId,
    ) -> None:
        """Remove a relationship from the graph representation."""
        raise NotImplementedError