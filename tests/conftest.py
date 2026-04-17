"""
Pytest configuration and shared fixtures for FESTIM-NIUQ tests.
"""

import pytest


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
