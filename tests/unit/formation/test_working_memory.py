import pytest

from memory_engine.formation.episodes import EpisodeManager
from memory_engine.formation.ingestion import IngestionService
from memory_engine.formation.working_memory import (
    WorkingMemoryManager,
    WorkingMemoryStatus,
)


@pytest.fixture
def setup():
    ingestion = IngestionService()
    episode_manager = EpisodeManager()
    working_memory = WorkingMemoryManager(capacity=3)

    event = ingestion.ingest(
        source="chat",
        content="Learning PyTorch.",
    )
    message = episode_manager.add_event(event, speaker="user")

    return working_memory, message


def test_working_memory_starts_empty():
    wm = WorkingMemoryManager(capacity=5)

    assert len(wm) == 0
    assert wm.capacity == 5


def test_invalid_capacity_is_rejected():
    with pytest.raises(ValueError):
        WorkingMemoryManager(capacity=0)


def test_non_integer_capacity_is_rejected():
    with pytest.raises(TypeError):
        WorkingMemoryManager(capacity=2.5)


def test_insert_and_retrieve(setup):
    wm, message = setup

    entry = wm.insert(message, priority=0.8)

    assert entry.message_id == message.message_id
    assert entry.priority == 0.8
    assert wm.retrieve(message.message_id) == entry
    assert len(wm) == 1


def test_duplicate_message_is_rejected(setup):
    wm, message = setup

    wm.insert(message)

    with pytest.raises(ValueError, match="already exists"):
        wm.insert(message)


def test_capacity_limit_is_enforced():
    ingestion = IngestionService()
    episode_manager = EpisodeManager()
    wm = WorkingMemoryManager(capacity=1)

    event1 = ingestion.ingest(source="chat", content="First")
    message1 = episode_manager.add_event(event1, speaker="user")
    wm.insert(message1)

    event2 = ingestion.ingest(source="chat", content="Second")
    message2 = episode_manager.add_event(event2, speaker="user")

    with pytest.raises(OverflowError, match="capacity reached"):
        wm.insert(message2)


def test_update_priority(setup):
    wm, message = setup
    wm.insert(message, priority=0.5)

    updated = wm.update(message.message_id, priority=0.9)

    assert updated.priority == 0.9


def test_update_status(setup):
    wm, message = setup
    wm.insert(message)

    updated = wm.promote(message.message_id)

    assert updated.status == WorkingMemoryStatus.PROMOTED


def test_promote_unknown_message_fails():
    from memory_engine.kernel.identity.ids import generate_message_id

    wm = WorkingMemoryManager()

    with pytest.raises(KeyError):
        wm.promote(generate_message_id())


def test_demote_reduces_priority(setup):
    wm, message = setup
    wm.insert(message, priority=0.8)

    demoted = wm.demote(message.message_id)

    assert demoted.priority == pytest.approx(0.4)
    assert demoted.status == WorkingMemoryStatus.DEMOTED


def test_remove_entry(setup):
    wm, message = setup
    wm.insert(message)

    removed = wm.remove(message.message_id)

    assert removed.message_id == message.message_id
    assert wm.retrieve(message.message_id) is None
    assert len(wm) == 0


def test_remove_unknown_message_fails(setup):
    from memory_engine.kernel.identity.ids import generate_message_id

    wm, _ = setup

    with pytest.raises(KeyError):
        wm.remove(generate_message_id())


def test_compress_removes_lowest_priority_first():
    ingestion = IngestionService()
    episode_manager = EpisodeManager()
    wm = WorkingMemoryManager(capacity=5)

    messages = []

    for content in ["Low", "Medium", "High"]:
        event = ingestion.ingest(source="chat", content=content)
        messages.append(
            episode_manager.add_event(event, speaker="user")
        )

    wm.insert(messages[0], priority=0.1)
    wm.insert(messages[1], priority=0.5)
    wm.insert(messages[2], priority=0.9)

    removed = wm.compress(target_size=2)

    assert len(removed) == 1
    assert removed[0].message.content == "Low"
    assert len(wm) == 2
    assert wm.retrieve(messages[2].message_id) is not None


def test_compress_rejects_invalid_target_size():
    wm = WorkingMemoryManager(capacity=3)

    with pytest.raises(ValueError):
        wm.compress(target_size=-1)

    with pytest.raises(ValueError):
        wm.compress(target_size=4)


def test_compress_does_not_remove_items_when_target_is_large(setup):
    wm, message = setup
    wm.insert(message)

    removed = wm.compress(target_size=2)

    assert removed == ()
    assert len(wm) == 1


def test_snapshot_is_read_only(setup):
    wm, message = setup
    wm.insert(message)

    snapshot = wm.snapshot()

    with pytest.raises(TypeError):
        snapshot[message.message_id] = snapshot[message.message_id]

def test_clear_removes_all_entries(setup):
    wm, message = setup
    wm.insert(message)

    removed = wm.clear()

    assert len(removed) == 1
    assert len(wm) == 0


def test_invalid_priority_is_rejected(setup):
    wm, message = setup

    with pytest.raises(ValueError):
        wm.insert(message, priority=1.5)
def test_compress_preserves_promoted_entries():
    ingestion = IngestionService()
    episode_manager = EpisodeManager()
    wm = WorkingMemoryManager(capacity=3)

    messages = []

    for content in ["Low priority", "Promoted", "High priority"]:
        event = ingestion.ingest(
            source="chat",
            content=content,
        )
        messages.append(
            episode_manager.add_event(event, speaker="user")
        )

    wm.insert(messages[0], priority=0.1)
    wm.insert(messages[1], priority=0.5)
    wm.insert(messages[2], priority=0.9)

    wm.promote(messages[1].message_id)

    removed = wm.compress(target_size=2)

    assert len(removed) == 1
    assert removed[0].message_id == messages[0].message_id
    assert wm.retrieve(messages[1].message_id) is not None
    assert wm.retrieve(messages[2].message_id) is not None


def test_compress_rejects_target_smaller_than_promoted_entries():
    ingestion = IngestionService()
    episode_manager = EpisodeManager()
    wm = WorkingMemoryManager(capacity=3)

    messages = []

    for content in ["First", "Second"]:
        event = ingestion.ingest(
            source="chat",
            content=content,
        )
        messages.append(
            episode_manager.add_event(event, speaker="user")
        )

    for message in messages:
        wm.insert(message)
        wm.promote(message.message_id)

    with pytest.raises(ValueError, match="promoted entries"):
        wm.compress(target_size=1)

    assert len(wm) == 2

