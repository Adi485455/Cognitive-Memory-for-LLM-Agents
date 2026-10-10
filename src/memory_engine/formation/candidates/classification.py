from dataclasses import dataclass
import re
from memory_engine.kernel.models.memory_type import MemoryType


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    memory_type: MemoryType
    confidence: float
    matched_rules: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.memory_type, MemoryType):
            raise TypeError("memory_type must be a MemoryType")

        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise TypeError("confidence must be a number")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        if not isinstance(self.matched_rules, tuple):
            raise TypeError("matched_rules must be a tuple")

        if any(not isinstance(rule, str) or not rule.strip()
               for rule in self.matched_rules):
            raise ValueError("matched_rules must contain non-empty strings")
        
class MemoryTypeClassifier:
    """Classifies memory text using transparent heuristic rules."""

    _RULES = (
        (
            MemoryType.PREFERENCE,
            0.90,
            "preference_expression",
            re.compile(
                r"\b(i prefer|i like|i dislike|i hate|"
                r"my preference|i would rather)\b",
                re.IGNORECASE,
            ),
        ),
        (
            MemoryType.PROCEDURAL,
            0.85,
            "procedural_instruction",
            re.compile(
                r"\b(step \d+|first,|then,|finally,|"
                r"you should|you must|make sure to|"
                r"run the tests|before deploying)\b",
                re.IGNORECASE,
            ),
        ),
            (
            MemoryType.EPISODIC,
            0.80,
            "past_experience",
            re.compile(
                r"\b(i attended|i visited|i experienced|"
                r"i completed|i met|i went to|"
                r"yesterday i|last week i|last month i)\b",
                re.IGNORECASE,
            ),
        ),
        (
            MemoryType.TEMPORAL,
            0.85,
            "temporal_expression",
            re.compile(
                r"\b(today|tomorrow|yesterday|next week|"
                r"next month|last week|scheduled for|"
                r"on monday|on tuesday|on wednesday|"
                r"on thursday|on friday|on saturday|"
                r"on sunday|at \d{1,2}(:\d{2})?\s*(am|pm)?)\b"
                r"|\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b",
                re.IGNORECASE,
            ),
        ),

        (
            MemoryType.WORKING,
            0.75,
            "current_task_context",
            re.compile(
                r"\b(i am currently|i'm currently|"
                r"right now i am|right now i'm|"
                r"currently debugging|currently working on|"
                r"currently building)\b",
                re.IGNORECASE,
            ),
        ),
    )

    def classify(self, content: str) -> ClassificationResult:
        if not isinstance(content, str):
            raise TypeError("content must be a string")

        if not content.strip():
            raise ValueError("content cannot be empty")

        for memory_type, confidence, rule_name, pattern in self._RULES:
            if pattern.search(content):
                return ClassificationResult(
                    memory_type=memory_type,
                    confidence=confidence,
                    matched_rules=(rule_name,),
                )

        return ClassificationResult(
            memory_type=MemoryType.SEMANTIC,
            confidence=0.50,
            matched_rules=("default_fallback",),
        )
        