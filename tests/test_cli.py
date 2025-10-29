from typer.testing import CliRunner

from healpix_painter.cli import app

runner = CliRunner()


def test_cli_default():
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
