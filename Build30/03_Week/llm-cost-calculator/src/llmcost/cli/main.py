"""Typer CLI."""

import typer

app = typer.Typer(help="LLM cost calculator.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """LLM cost calculator."""
