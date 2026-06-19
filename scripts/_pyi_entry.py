"""PyInstaller entry point — bundles the agent into a single executable.

The packaged binary behaves exactly like ``./cadaXXXX_alpha`` (it just doesn't
need a system Python / uv venv).  abc and yosys are still external programs,
resolved at runtime via configs/tools.yaml (next to the binary), the -config
file's ``tools:`` section, env vars, or $PATH.
"""

import sys

from cada.main import main

if __name__ == "__main__":
    sys.exit(main())
