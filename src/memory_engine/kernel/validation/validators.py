from memory_engine.kernel.models.memory import Memory
from memory_engine.kernel.validation.invariants import (
    validate_memory_invariants,
)


def validate_memory(memory: Memory) -> None:
    """Validate a memory before it enters canonical state."""
    validate_memory_invariants(memory)