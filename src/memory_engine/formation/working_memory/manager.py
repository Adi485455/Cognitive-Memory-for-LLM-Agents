from threading import RLock
from types import MappingProxyType
from typing import Mapping

from memory_engine.kernel.identity.ids import MessageId
from memory_engine.kernel.models.message import Message
from memory_engine.formation.working_memory.models import (
    WorkingMemoryEntry,
    WorkingMemoryStatus,
)


class WorkingMemoryManager:
    """
    Manages a bounded, process-local working set of messages.

    This implementation does not persist data and does not create
    canonical memories.
    """

    def __init__(self, capacity: int = 100) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")

        if capacity < 1:
            raise ValueError("capacity must be at least 1")

        self._capacity = capacity
        self._entries: dict[MessageId, WorkingMemoryEntry] = {}
        self._lock = RLock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def insert(
        self,
        message: Message,
        *,
        priority: float = 0.5,
    ) -> WorkingMemoryEntry:
        """Insert a message into the working set."""
        entry = WorkingMemoryEntry(
            message=message,
            priority=priority,
        )

        with self._lock:
            if message.message_id in self._entries:
                raise ValueError("Message already exists in working memory")

            if len(self._entries) >= self._capacity:
                raise OverflowError("Working memory capacity reached")

            self._entries[message.message_id] = entry

        return entry

    def update(
        self,
        message_id: MessageId,
        *,
        priority: float | None = None,
        status: WorkingMemoryStatus | None = None,
    ) -> WorkingMemoryEntry:
        """Update an entry's priority and/or processing status."""
        with self._lock:
            current = self._entries.get(message_id)

            if current is None:
                raise KeyError(f"Unknown message ID: {message_id}")

            updated = WorkingMemoryEntry(
                message=current.message,
                priority=(
                    current.priority if priority is None else priority
                ),
                status=current.status if status is None else status,
                inserted_at=current.inserted_at,
            )

            self._entries[message_id] = updated
            return updated

    def remove(self, message_id: MessageId) -> WorkingMemoryEntry:
        """Remove an entry from the active working set."""
        with self._lock:
            try:
                return self._entries.pop(message_id)
            except KeyError:
                raise KeyError(
                    f"Unknown message ID: {message_id}"
                ) from None

    def retrieve(self, message_id: MessageId) -> WorkingMemoryEntry | None:
        """Retrieve one entry by message ID."""
        with self._lock:
            return self._entries.get(message_id)

    def snapshot(self) -> Mapping[MessageId, WorkingMemoryEntry]:
        """Return a read-only snapshot of the current entries."""
        with self._lock:
            return MappingProxyType(dict(self._entries))
    def compress(self, target_size: int) -> tuple[WorkingMemoryEntry, ...]:
        if isinstance(target_size, bool) or not isinstance(target_size, int):
            raise TypeError("target_size must be an integer")

        if target_size < 0 or target_size > self._capacity:
            raise ValueError(
                f"target_size must be between 0 and {self._capacity}"
            )

        with self._lock:
            if target_size >= len(self._entries):
                return ()

            protected = [
                entry
                for entry in self._entries.values()
                if entry.status == WorkingMemoryStatus.PROMOTED
            ]

            if len(protected) > target_size:
                raise ValueError(
                    "target_size cannot be smaller than the number "
                    "of promoted entries"
                )

            removable = sorted(
                (
                    entry
                    for entry in self._entries.values()
                    if entry.status != WorkingMemoryStatus.PROMOTED
                ),
                key=lambda entry: (
                    entry.priority,
                    entry.inserted_at,
                    str(entry.message_id),
                ),
            )

            remove_count = len(self._entries) - target_size

            if len(removable) < remove_count:
                raise ValueError(
                    "Cannot reach target_size without removing promoted entries"
                )

            removed = removable[:remove_count]

            for entry in removed:
                del self._entries[entry.message_id]

            return tuple(removed)



    def promote(self, message_id: MessageId) -> WorkingMemoryEntry:
        """Mark an entry for the next memory-formation stage."""
        return self.update(
            message_id,
            status=WorkingMemoryStatus.PROMOTED,
        )

    def demote(self, message_id: MessageId) -> WorkingMemoryEntry:
        """Mark an entry as lower-priority processing context."""
        with self._lock:
            current = self._entries.get(message_id)

            if current is None:
                raise KeyError(f"Unknown message ID: {message_id}")

            return self.update(
                message_id,
                priority=current.priority * 0.5,
                status=WorkingMemoryStatus.DEMOTED,
            )

    def clear(self) -> tuple[WorkingMemoryEntry, ...]:
        """Remove all entries and return the removed entries."""
        with self._lock:
            removed = tuple(self._entries.values())
            self._entries.clear()
            return removed

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)