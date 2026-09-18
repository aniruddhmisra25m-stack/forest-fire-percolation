import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run the slow literature-recovery tests (~5 min)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow statistical test, run with --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
