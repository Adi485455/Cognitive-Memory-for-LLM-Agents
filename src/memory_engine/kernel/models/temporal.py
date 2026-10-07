#created_at    → when the memory record was created
#learned_at    → when the system learned it
#event_time    → when the actual event happened
#valid_from    → when the information became valid
#valid_until   → when the information stopped being valid




from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Temporal:
    created_at: datetime
    learned_at: datetime
    event_time: datetime | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    def __post_init__(self) -> None:
        if self.valid_from is not None and self.valid_until is not None:
            if self.valid_until < self.valid_from:
                raise ValueError(
                    "valid_until cannot be earlier than valid_from"
                )