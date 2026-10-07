from dataclasses import dataclass


@dataclass(frozen=True)
class Version:
    number: int

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("Version number must be greater than or equal to 1")