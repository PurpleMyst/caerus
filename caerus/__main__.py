from __future__ import annotations

import typing as t

import click

from .cli import CLI
from .utils import FFMpeg


@click.group()
@click.option("-d", "--database", type=click.Path(), default="database.db")
@click.pass_context
def cli(ctx: click.Context, database: str) -> None:
    ctx.ensure_object(dict)
    ctx.obj["cli"] = CLI(database)


@cli.group()
def references() -> None:
    """Manipulate segment references."""
    pass


@references.command()
@click.argument("path", type=click.Path())
@click.option("-s", "--start", type=float, required=True)
@click.option("-e", "--end", type=float)
@click.option("-d", "--description", type=str, required=True)
@click.pass_context
def add(
    ctx: click.Context,
    path: str,
    start: float,
    end: t.Optional[float],
    description: str,
) -> None:
    """Define a new segment reference in PATH."""
    cli: CLI = ctx.obj["cli"]
    cli.add_reference(path, description, start, end)


@references.command()
@click.argument("description", type=str)
@click.pass_context
def remove(
    ctx: click.Context,
    description: str,
) -> None:
    """Remove a segment reference with the given DESCRIPTION."""
    cli: CLI = ctx.obj["cli"]
    cli.remove_reference(description)


@references.command()
@click.argument("path", type=click.Path())
@click.option(
    "--all-in-series/--file-only",
    type=bool,
    help="Show all segments in the file's series "
    "or only segments defined for the specific file.",
    default=True,
)
@click.pass_context
def show(ctx: click.Context, path: str, all_in_series: bool) -> None:
    """Show segment references defined on a file, potentially on all of its series"""
    cli: CLI = ctx.obj["cli"]
    cli.show_references(path, all_in_series)


@cli.command()
@click.argument("path", type=click.Path())
@click.option("-o", "--output", type=click.Path(), default="out.mp4")
@click.option("-f", "--ffmpeg-arg", type=str, nargs=2, multiple=True)
@click.pass_context
def shave(
    ctx: click.Context, path: str, output: str, ffmpeg_arg: t.List[t.Tuple[str, str]]
) -> None:
    """Remove segments found in the given video file, saving the shaved version to OUTPUT.

    This is done by searching for known segments inside PATH"""
    cli: CLI = ctx.obj["cli"]
    cli.shave(path, output, FFMpeg(dict(ffmpeg_arg)))


if __name__ == "__main__":
    cli()
