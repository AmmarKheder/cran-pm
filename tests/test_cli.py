"""Tests for the cranpm CLI entry point."""

import pytest

from cranpm.cli.main import build_parser, main


def test_parser_builds():
    parser = build_parser()
    assert parser.prog == "cranpm"


def test_no_args_prints_help(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cranpm" in out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_unimplemented_subcommand(capsys):
    """`train` and `download` are still placeholders; the CLI must report so."""
    rc = main(["train"])
    assert rc == 1
    assert "not implemented yet" in capsys.readouterr().out


def test_forecast_subcommand_requires_args(capsys):
    """`forecast` is wired but requires --checkpoint, --inputs, --output."""
    with pytest.raises(SystemExit) as excinfo:
        main(["forecast"])
    assert excinfo.value.code == 2  # argparse error
