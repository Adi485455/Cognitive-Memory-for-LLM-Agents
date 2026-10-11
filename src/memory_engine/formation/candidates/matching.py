import re
from dataclasses import dataclass

from memory_engine.formation.candidates.models import CandidateMemory
from memory_engine.kernel.models.memory import Memory


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    """A potential match between a candidate and an existing memory."""

    memory_id: object
    score: float
    match_type: str

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(
            self.score, (int, float)
        ):
            raise TypeError("score must be numeric")

        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")

        if not isinstance(self.match_type, str) or not self.match_type.strip():
            raise ValueError("match_type must be a non-empty string")


class CandidateMemoryMatcher:
    """Finds potentially related memories using lexical similarity."""

    def __init__(self, min_score: float = 0.30) -> None:
        if isinstance(min_score, bool) or not isinstance(
            min_score, (int, float)
        ):
            raise TypeError("min_score must be numeric")

        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0")

        self._min_score = float(min_score)

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.casefold()
        text = re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(text.split())

    def _compare(self, candidate_content: str, memory_content: str) -> tuple[float, str]:
        candidate_normalized = self._normalize(candidate_content)
        memory_normalized = self._normalize(memory_content)

        if candidate_normalized == memory_normalized:
            return 1.0, "normalized_exact"

        candidate_tokens = self._tokens(candidate_normalized)
        memory_tokens = self._tokens(memory_normalized)

        if not candidate_tokens or not memory_tokens:
            return 0.0, "no_overlap"

        intersection = candidate_tokens & memory_tokens
        union = candidate_tokens | memory_tokens

        score = len(intersection) / len(union)

        return score, "token_jaccard"

    def find_matches(
        self,
        candidate: CandidateMemory,
        memories: list[Memory] | tuple[Memory, ...],
    ) -> tuple[MemoryMatch, ...]:
        if not isinstance(candidate, CandidateMemory):
            raise TypeError("candidate must be a CandidateMemory instance")

        if not isinstance(memories, (list, tuple)):
            raise TypeError("memories must be a list or tuple")

        matches: list[MemoryMatch] = []

        for memory in memories:
            if not isinstance(memory, Memory):
                raise TypeError("every existing item must be a Memory")

            score, match_type = self._compare(
                candidate.content,
                memory.content,
            )

            if score >= self._min_score:
                matches.append(
                    MemoryMatch(
                        memory_id=memory.memory_id,
                        score=score,
                        match_type=match_type,
                    )
                )

        matches.sort(key=lambda match: match.score, reverse=True)

        return tuple(matches)