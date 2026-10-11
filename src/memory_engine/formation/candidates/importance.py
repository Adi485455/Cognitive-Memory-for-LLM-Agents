import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImportanceResult:
    """Result of candidate importance scoring."""

    score: float
    matched_rules: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(
            self.score, (int, float)
        ):
            raise TypeError("score must be numeric")

        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")

        object.__setattr__(self, "score", float(self.score))

        if not isinstance(self.matched_rules, tuple):
            raise TypeError("matched_rules must be a tuple")

        if any(
            not isinstance(rule, str) or not rule.strip()
            for rule in self.matched_rules
        ):
            raise ValueError(
                "matched_rules must contain non-empty strings"
            )


class CandidateImportanceScorer:
    """Estimates candidate importance using transparent rules."""

    _RULES = (
        (
            "explicit_preference",
            re.compile(
                r"\b(i prefer|i like|i dislike|i hate|"
                r"my preference|i would rather)\b",
                re.IGNORECASE,
            ),
            0.75,
        ),
        (
            "long_term_goal",
            re.compile(
                r"\b(i want to learn|my goal is|"
                r"i aim to|i plan to become|"
                r"i am preparing for)\b",
                re.IGNORECASE,
            ),
            0.80,
        ),
        (
            "ongoing_project",
            re.compile(
                r"\b(currently building|currently working on|"
                r"i am developing|i'm developing|"
                r"my project is)\b",
                re.IGNORECASE,
            ),
            0.75,
        ),
        (
            "personal_fact",
            re.compile(
                r"\b(my name is|i live in|"
                r"i work as|i study at|"
                r"i am studying)\b",
                re.IGNORECASE,
            ),
            0.70,
        ),
        (
            "temporary_event",
            re.compile(
                r"\b(i had lunch|i ate lunch|"
                r"i went shopping|i watched a movie)\b",
                re.IGNORECASE,
            ),
            0.20,
        ),
    )

    def score(self, content: str) -> ImportanceResult:
        """Return an importance estimate and its supporting rules."""

        if not isinstance(content, str):
            raise TypeError("content must be a string")

        if not content.strip():
            raise ValueError("content must not be empty")

        matched_rules = []
        matched_scores = []

        for rule_name, pattern, rule_score in self._RULES:
            if pattern.search(content):
                matched_rules.append(rule_name)
                matched_scores.append(rule_score)

        if not matched_scores:
            return ImportanceResult(
                score=0.50,
                matched_rules=("default_fallback",),
            )

        # Use the strongest matching signal rather than adding scores
        # together and accidentally inflating importance.
        return ImportanceResult(
            score=max(matched_scores),
            matched_rules=tuple(matched_rules),
        )