# Copyright (C) 2024 Tadej Novak
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0

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
