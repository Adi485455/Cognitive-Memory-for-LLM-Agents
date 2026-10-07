from dataclasses import dataclass


@dataclass(frozen=True)
class Reinforcement:
    support_count: int = 0
    confirmation_count: int = 0
    reinforcement_score: float = 0.0
    reinforcement_reason: str | None = None

    def __post_init__(self) -> None:
        if self.support_count < 0:
            raise ValueError("support_count cannot be negative")

        if self.confirmation_count < 0:
            raise ValueError("confirmation_count cannot be negative")

        if not 0.0 <= self.reinforcement_score <= 1.0:
            raise ValueError(
                "reinforcement_score must be between 0.0 and 1.0"
            )