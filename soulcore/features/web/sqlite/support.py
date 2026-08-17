from __future__ import annotations

import hashlib as hashlib
import math as math
import sqlite3 as sqlite3
from collections.abc import Mapping as Mapping
from datetime import datetime as datetime
from datetime import timedelta as timedelta
from typing import Any as Any

from ....contracts.models import RoleProfile as RoleProfile
from ....contracts.web import (
    WebCallerKind as WebCallerKind,
)
from ....contracts.web import (
    WebReadStatus as WebReadStatus,
)
from ....contracts.web import (
    WebSearchIntensity as WebSearchIntensity,
)
from ....contracts.web import (
    WebSearchPurpose as WebSearchPurpose,
)
from ....storage.sqlite.codec import (
    _coerce_datetime as _coerce_datetime,
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
from ..domain import (
    WebImageSearchResultRecord as WebImageSearchResultRecord,
)
from ..domain import (
    WebPageSnapshotRecord as WebPageSnapshotRecord,
)
from ..domain import (
    WebSearchKind as WebSearchKind,
)
from ..domain import (
    WebSearchProviderRecord as WebSearchProviderRecord,
)
from ..domain import (
    WebSearchResultRecord as WebSearchResultRecord,
)
from ..domain import (
    WebSearchSessionRecord as WebSearchSessionRecord,
)
from ..domain import (
    WebSearchSessionStatus as WebSearchSessionStatus,
)

__all__ = [name for name in globals() if not name.startswith("__")]
