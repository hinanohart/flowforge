"""Tests for the click CLI."""

from click.testing import CliRunner

from flowforge.cli.main import cli


def test_version_flag():
    runner = CliRunner()
    res = runner.invoke(cli, ["--version"])
    assert res.exit_code == 0
    assert "flowforge" in res.output


def test_help():
    runner = CliRunner()
    res = runner.invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "run" in res.output
    assert "auto" in res.output
    assert "status" in res.output


def test_run_dry_run():
    runner = CliRunner()
    res = runner.invoke(
        cli, ["run", "--dry-run", "--task", "libero_spatial", "--gens", "1", "--pop", "2"]
    )
    assert res.exit_code == 0
    assert "dry-run" in res.output


def test_status_without_state(tmp_path):
    runner = CliRunner()
    res = runner.invoke(cli, ["status", "--root", str(tmp_path)])
    assert res.exit_code == 0
    assert "no state.json" in res.output


def test_auto_background_rejected(tmp_path):
    runner = CliRunner()
    res = runner.invoke(cli, ["auto", "--background", "--root", str(tmp_path)])
    assert res.exit_code != 0
    assert "WSL2" in res.output or "WSL2" in str(res.exception)
