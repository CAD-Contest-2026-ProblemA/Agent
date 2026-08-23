"""PyInstaller entry point — bundles the agent into a single executable.

The packaged binary does not need a system Python / uv venv.  Its routing
default is selected by the marker added in ``scripts/build.sh`` (pure LLM by
default); ``--rules`` and ``--no-rules`` remain runtime overrides.  abc and
yosys are still external programs, resolved at runtime via configs/tools.yaml
(next to the binary), the -config file's ``tools:`` section, env vars, or PATH.
"""

import sys

from cada.main import main

if __name__ == "__main__":
    sys.exit(main())
