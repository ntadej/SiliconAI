"""Tests configuration."""
from os import environ

import pytest


@pytest.fixture(autouse=True)
def env() -> dict[str, str]:
    """Return environment for tests."""
    values: dict[str, str] = {
        "SILICONAI_GLOBAL_CONFIG": "tests/resources/config.toml",
    }
    for key, value in values.items():
        environ[key] = value
    return values
