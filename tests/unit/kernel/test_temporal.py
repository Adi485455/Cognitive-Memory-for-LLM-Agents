from datetime import datetime, timezone

import pytest

from memory_engine.kernel.models.temporal import Temporal


def test_temporal_can_be_created():
    created_at = datetime.now(timezone.utc)
    learned_at = datetime.now(timezone.utc)

    temporal = Temporal(
        created_at=created_at,
        learned_at=learned_at,
    )

    assert temporal.created_at == created_at
    assert temporal.learned_at == learned_at
    assert temporal.event_time is None
    assert temporal.valid_from is None
    assert temporal.valid_until is None


def test_temporal_can_store_full_time_information():
    created_at = datetime(2026, 10, 7, tzinfo=timezone.utc)
    learned_at = datetime(2026, 10, 7, tzinfo=timezone.utc)
    event_time = datetime(2026, 10, 1, tzinfo=timezone.utc)
    valid_from = datetime(2026, 10, 1, tzinfo=timezone.utc)
    valid_until = datetime(2026, 12, 31, tzinfo=timezone.utc)

    temporal = Temporal(
        created_at=created_at,
        learned_at=learned_at,
        event_time=event_time,
        valid_from=valid_from,
        valid_until=valid_until,
    )

    assert temporal.event_time == event_time
    assert temporal.valid_from == valid_from
    assert temporal.valid_until == valid_until


def test_valid_until_cannot_be_before_valid_from():
    valid_from = datetime(2026, 10, 10, tzinfo=timezone.utc)
    valid_until = datetime(2026, 10, 1, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        Temporal(
            created_at=valid_from,
            learned_at=valid_from,
            valid_from=valid_from,
            valid_until=valid_until,
        )