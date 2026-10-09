from datetime import datetime, timedelta, timezone

import pytest

from memory_engine.kernel.models.episode import Episode


def test_episode_creation():
    episode = Episode(source="chat")

    assert episode.source == "chat"
    assert episode.episode_id is not None
    assert episode.started_at.tzinfo is not None
    assert episode.ended_at is None


def test_episode_ids_are_unique():
    first = Episode(source="chat")
    second = Episode(source="chat")

    assert first.episode_id != second.episode_id


def test_episode_normalizes_source():
    episode = Episode(source="  chat  ")

    assert episode.source == "chat"


def test_episode_rejects_empty_source():
    with pytest.raises(ValueError):
        Episode(source="  ")


def test_episode_rejects_naive_timestamp():
    with pytest.raises(ValueError):
        Episode(
            source="chat",
            started_at=datetime(2026, 10, 9, 10, 0),
        )


def test_episode_rejects_end_before_start():
    start = datetime(2026, 10, 9, 10, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError):
        Episode(
            source="chat",
            started_at=start,
            ended_at=start - timedelta(minutes=1),
        )


def test_episode_metadata_is_copied():
    metadata = {"project": "memory"}
    episode = Episode(source="chat", metadata=metadata)

    metadata["project"] = "changed"

    assert episode.metadata["project"] == "memory"