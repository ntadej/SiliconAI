"""Main CLI tests."""
from pathlib import Path

import pytest
from typer.testing import CliRunner

runner = CliRunner()


@pytest.mark.forked()
def test_help(env: dict[str, str]) -> None:
    """Test help."""
    from siliconai.cli import application

    result = runner.invoke(application, ["--help"], env=env, catch_exceptions=False)
    assert result.exit_code == 0


@pytest.mark.forked()
def test_config_missing(env: dict[str, str]) -> None:
    """Test config missing."""
    from siliconai.cli import application

    env["SILICONAI_GLOBAL_CONFIG"] = "tests/resources/missing.toml"

    result = runner.invoke(application, ["config"], env=env, catch_exceptions=False)
    assert result.exit_code == 1


@pytest.mark.forked()
def test_version(env: dict[str, str]) -> None:
    """Test version."""
    from siliconai.cli import application

    result = runner.invoke(application, ["--version"], env=env, catch_exceptions=False)
    assert result.exit_code == 0


@pytest.mark.forked()
def test_config_generate(env: dict[str, str]) -> None:
    """Test config generation."""
    from siliconai.cli import application

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


@pytest.mark.forked()
def test_config(env: dict[str, str]) -> None:
    """Test config."""
    from siliconai.cli import application

    result = runner.invoke(application, ["config"], env=env, catch_exceptions=False)
    assert result.exit_code == 0
