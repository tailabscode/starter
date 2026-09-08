"""Allows `python -m todo_cli ...` in addition to the `todo` console script."""

from todo_cli.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
