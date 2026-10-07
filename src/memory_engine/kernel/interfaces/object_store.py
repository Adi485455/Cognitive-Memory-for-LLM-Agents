from abc import ABC, abstractmethod


class ObjectStore(ABC):

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        """Store raw object data under a stable key."""
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Retrieve raw object data by key."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete an object by key."""
        raise NotImplementedError