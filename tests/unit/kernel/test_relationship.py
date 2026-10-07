from datetime import datetime, timezone

import pytest

from memory_engine.kernel.identity.ids import (
    generate_entity_id,
    generate_relationship_id,
)
from memory_engine.kernel.models.relationship import Relationship


def test_relationship_can_be_created():
    relationship_id = generate_relationship_id()
    source_id = generate_entity_id()
    target_id = generate_entity_id()

    relationship = Relationship(
        relationship_id=relationship_id,
        source_id=source_id,
        target_id=target_id,
        relationship_type="RELATED_TO",
        confidence=0.9,
        inference_type="explicit",
    )

    assert relationship.relationship_id == relationship_id
    assert relationship.source_id == source_id
    assert relationship.target_id == target_id
    assert relationship.relationship_type == "RELATED_TO"
    assert relationship.confidence == 0.9
    assert relationship.inference_type == "explicit"


def test_relationship_can_have_temporal_validity():
    relationship_id = generate_relationship_id()
    source_id = generate_entity_id()
    target_id = generate_entity_id()

    valid_from = datetime(2026, 1, 1, tzinfo=timezone.utc)
    valid_until = datetime(2026, 12, 31, tzinfo=timezone.utc)

    relationship = Relationship(
        relationship_id=relationship_id,
        source_id=source_id,
        target_id=target_id,
        relationship_type="USES",
        confidence=0.95,
        inference_type="explicit",
        valid_from=valid_from,
        valid_until=valid_until,
    )

    assert relationship.valid_from == valid_from
    assert relationship.valid_until == valid_until


def test_relationship_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        Relationship(
            relationship_id=generate_relationship_id(),
            source_id=generate_entity_id(),
            target_id=generate_entity_id(),
            relationship_type="RELATED_TO",
            confidence=1.5,
            inference_type="derived",
        )


def test_relationship_rejects_invalid_temporal_range():
    valid_from = datetime(2026, 12, 31, tzinfo=timezone.utc)
    valid_until = datetime(2026, 1, 1, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        Relationship(
            relationship_id=generate_relationship_id(),
            source_id=generate_entity_id(),
            target_id=generate_entity_id(),
            relationship_type="RELATED_TO",
            confidence=0.8,
            inference_type="derived",
            valid_from=valid_from,
            valid_until=valid_until,
        )