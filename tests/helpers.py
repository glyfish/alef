"""Small typing helpers shared across the alef test suite.

navi's report models make several fields ``Optional`` because
``from_dict``/``__init__`` legitimately leave them empty -- a deserialised
``StatisticalTestData`` with no ``sig`` key, an ``OLSResult`` before
``set_transforms`` has run. The code paths exercised here always populate them,
but the type checker cannot know that, so reads like ``data.sig.label`` look
like attribute access on ``None``.
"""
from typing import TypeVar

T = TypeVar("T")


def present(value: T | None) -> T:
    """Narrow an Optional model field the caller knows is populated.

    The assertion never fires in a passing test; it is here so that a
    regression which *does* leave the field empty fails at the point of the
    read with a clear message instead of an ``AttributeError`` on ``None``.
    """
    assert value is not None
    return value
