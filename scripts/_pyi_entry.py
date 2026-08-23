"""PyInstaller entry point — bundles the agent into a single executable.

The packaged binary does not need a system Python / uv venv.  Its routing and
BM25 defaults are selected by markers added in ``scripts/build.sh`` (pure LLM
with BM25 by default); ``--rules`` / ``--no-rules`` and ``--bm25`` /
``--no-bm25`` remain runtime overrides.  abc and yosys are still external,
resolved via configs/tools.yaml, the -config ``tools:`` section, env, or PATH.
"""

import sys

from cada.main import main

if __name__ == "__main__":
    sys.exit(main())
