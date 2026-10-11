import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfidenceResult:
    """Result of candidate confidence scoring."""

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


class CandidateConfidenceScorer:
    """Estimates confidence using transparent linguistic heuristics."""

    _RULES = (
        (
            "explicit_uncertainty",
            re.compile(
                r"\b(i think|i guess|maybe|perhaps|"
                r"i'm not sure|i am not sure|"
                r"might be|could be|possibly)\b",
                re.IGNORECASE,
            ),
            0.35,
        ),
        (
            "explicit_personal_experience",
            re.compile(
                r"\b(i attended|i visited|i completed|"
                r"i built|i developed|i implemented|"
                r"i experienced|i met)\b",
                re.IGNORECASE,
            ),
            0.75,
        ),
        (
            "explicit_preference",
            re.compile(
                r"\b(i prefer|i like|i dislike|"
                r"i hate|my preference)\b",
                re.IGNORECASE,
            ),
            0.70,
        ),
    )

    def score(self, content: str) -> ConfidenceResult:
        """Return a heuristic confidence estimate and matched rules."""

        if not isinstance(content, str):
            raise TypeError("content must be a string")

        if not content.strip():
            raise ValueError("content must not be empty")

        matched_rules = []

        for rule_name, pattern, _ in self._RULES:
            if pattern.search(content):
                matched_rules.append(rule_name)

        # Explicit uncertainty takes precedence over other signals.
        if "explicit_uncertainty" in matched_rules:
            score = 0.35
        elif "explicit_personal_experience" in matched_rules:
            score = 0.75
        elif "explicit_preference" in matched_rules:
            score = 0.70
        else:
            score = 0.50
            matched_rules.append("default_fallback")

        return ConfidenceResult(
            score=score,
            matched_rules=tuple(matched_rules),
        )