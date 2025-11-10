# Copyright (C) 2024 Tadej Novak
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0

"""Main CLI tests."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from siliconai.cli import application

runner = CliRunner()


@pytest.mark.forked
def test_help(env: dict[str, str]) -> None:
    """Test help."""
    result = runner.invoke(application, ["--help"], env=env, catch_exceptions=False)
    assert result.exit_code == 0


@pytest.mark.forked
def test_config_missing(env: dict[str, str]) -> None:
    """Test config missing."""
    env["SILICONAI_GLOBAL_CONFIG"] = "tests/resources/missing.toml"

    result = runner.invoke(application, ["config"], env=env, catch_exceptions=False)
    assert result.exit_code == 1


@pytest.mark.forked
def test_version(env: dict[str, str]) -> None:
    """Test version."""
    result = runner.invoke(application, ["--version"], env=env, catch_exceptions=False)
    assert result.exit_code == 0


@pytest.mark.forked
def test_config_generate(env: dict[str, str]) -> None:
    """Test config generation."""
    env["SILICONAI_GLOBAL_CONFIG"] = "tests/resources/missing.toml"

    result = runner.invoke(
        application,
        ["config", "--generate"],
        env=env,
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    config_path = Path(env["SILICONAI_GLOBAL_CONFIG"])
    assert config_path.exists()
    config_path.unlink()
    assert not config_path.exists()


@pytest.mark.forked
def test_config(env: dict[str, str]) -> None:
    """Test config."""
    result = runner.invoke(application, ["config"], env=env, catch_exceptions=False)
    assert result.exit_code == 0
