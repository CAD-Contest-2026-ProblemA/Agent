#!/usr/bin/env python3
"""Standalone launcher for the environment doctor.

    python3 scripts/doctor.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cada.doctor import main

if __name__ == "__main__":
    raise SystemExit(main())
