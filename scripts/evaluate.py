#!/usr/bin/env python3
"""Standalone launcher for the local evaluator.

    python3 scripts/evaluate.py [cases...] [--update-golden] [--verbose]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
