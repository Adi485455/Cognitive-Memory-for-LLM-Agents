import pytest

from memory_engine.kernel.models.version import Version


def test_initial_version_is_valid():
    version = Version(number=1)

    assert version.number == 1


def test_version_can_increment():
    version = Version(number=2)

    assert version.number == 2


def test_version_number_must_be_positive():
    with pytest.raises(ValueError):
        Version(number=0)

    with pytest.raises(ValueError):
        Version(number=-1)