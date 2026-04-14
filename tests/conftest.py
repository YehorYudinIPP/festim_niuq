"""
Pytest configuration and shared fixtures for FESTIM-NIUQ tests.
"""

import os
import sys
import pytest

# Ensure the repository root is on the Python path so that
# ``import uq.*`` and ``import festim_model.*`` work in tests
# regardless of where pytest is invoked from.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow scientific tests that require FESTIM",
    )


def pytest_collection_modifyitems(config, items):
    """Skip scientific tests unless --runslow is given."""
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="Need --runslow option to run")
        for item in items:
            if "scientific" in item.keywords:
                item.add_marker(skip_slow)
