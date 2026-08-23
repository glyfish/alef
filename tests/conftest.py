"""Shared fixtures for the alef test suite.

alef owns the tests for navi's analysis code (``lib/data``, ``lib/models``,
``lib/trading``) — navi carries no test suite of its own. ``lib`` is importable
here via navi's editable install (``-e ../navi``).

Every simulation in those modules draws from numpy's global RNG
(``numpy.random.normal`` etc.), so the autouse fixture below reseeds before
each test. A test that needs a different stream may call ``numpy.random.seed``
itself.
"""
import numpy
import pytest


@pytest.fixture(autouse=True)
def seed_numpy():
    """Make every test deterministic by resetting numpy's global RNG."""
    numpy.random.seed(20260820)
    yield
