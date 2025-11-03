# Test CLI call for basic job

from typer.testing import CliRunner

from healpix_painter.cli import app

runner = CliRunner()


def test_cli_default():
    """Call the CLI for a sample job, output to .cache in tests directory"""
    result = runner.invoke(
        app,
        [
            "--lvk-eventname",
            "S230922g",
            "--output-dir",
            "./.cache/test_cli_default",
        ],
    )
    assert result.exit_code == 0


def test_cli_update_tiling():
    """Call the CLI for a sample job, output to .cache in tests directory.
    Also updates the tiling cache.
    """
    result = runner.invoke(
        app,
        [
            "--lvk-eventname",
            "S230922g",
            "--output-dir",
            "./.cache/test_cli_update_tiling",
            "--tiling-force-update",
        ],
    )
    assert result.exit_code == 0
