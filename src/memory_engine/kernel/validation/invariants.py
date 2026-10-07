from memory_engine.kernel.models.memory import Memory


def validate_memory_invariants(memory: Memory) -> None:
    """Validate invariants required for canonical memory state."""

    if memory.memory_id is None:
        raise ValueError("Memory must have a stable memory_id")

    if memory.version.number < 1:
        raise ValueError("Memory version must be at least 1")

    if not memory.provenance.evidence_ids:
        raise ValueError(
            "Memory must have at least one provenance evidence reference"
        )

    if memory.temporal.valid_from is not None:
        if (
            memory.temporal.valid_until is not None
            and memory.temporal.valid_until < memory.temporal.valid_from
        ):
            raise ValueError(
                "valid_until cannot be earlier than valid_from"
            )