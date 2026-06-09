from __future__ import annotations

import os
import sys


if any(arg.endswith("pytest") or arg.endswith("pytest.exe") for arg in sys.argv[:1]):
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
