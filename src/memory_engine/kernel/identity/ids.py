from typing import NewType
from uuid import UUID, uuid4


MemoryId = NewType("MemoryId", UUID)
EpisodeId = NewType("EpisodeId", UUID)
MessageId = NewType("MessageId", UUID)
EvidenceId = NewType("EvidenceId", UUID)
EntityId = NewType("EntityId", UUID)
RelationshipId = NewType("RelationshipId", UUID)


def generate_memory_id() -> MemoryId:
    return MemoryId(uuid4())


def generate_episode_id() -> EpisodeId:
    return EpisodeId(uuid4())


def generate_message_id() -> MessageId:
    return MessageId(uuid4())


def generate_evidence_id() -> EvidenceId:
    return EvidenceId(uuid4())


def generate_entity_id() -> EntityId:
    return EntityId(uuid4())


def generate_relationship_id() -> RelationshipId:
    return RelationshipId(uuid4())