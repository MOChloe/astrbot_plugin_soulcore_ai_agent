from __future__ import annotations

import sqlite3 as sqlite3
from datetime import datetime as datetime
from datetime import timedelta as timedelta
from typing import Any as Any

from ....contracts.models import (
    ContextBuildReport as ContextBuildReport,
)
from ....contracts.models import (
    ConversationMessage as ConversationMessage,
)
from ....contracts.models import (
    DialogueSummary as DialogueSummary,
)
from ....contracts.models import (
    MessageDirection as MessageDirection,
)
from ....storage.sqlite.codec import (
    _dt as _dt,
)
from ....storage.sqlite.codec import (
    _dump as _dump,
)
from ....storage.sqlite.codec import (
    _load as _load,
)
from ....storage.sqlite.codec import (
    _now as _now,
)
from ....storage.sqlite.codec import (
    _parse as _parse,
)
from ....storage.sqlite.dialogue_turns import (
    CONTEXT_ELIGIBLE_INBOUND_STATUSES as CONTEXT_ELIGIBLE_INBOUND_STATUSES,
)
from ....storage.sqlite.dialogue_turns import context_eligible_sql as context_eligible_sql


def _context_eligible_sql() -> str:
    return context_eligible_sql()


__all__ = [name for name in globals() if not name.startswith("__")]
