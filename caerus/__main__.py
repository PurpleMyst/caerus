from __future__ import annotations

import typing as t

import click

from .cli import CLI
from .utils import FFMpeg


@click.group()
@click.option("-d", "--database", type=click.Path(), default="database.db")
@click.option("--preset", type=str, default="medium")
@click.option("--crf", type=int, default="23")
@click.pass_context
def cli(ctx: click.Context, database: str, preset: str, crf: int) -> None:
    ctx.ensure_object(dict)
    ctx.obj["cli"] = CLI(database, FFMpeg({"preset": preset, "crf": crf}))


@cli.command()
@click.argument("path", type=click.Path())
@click.option("-s", "--start", type=float, required=True)
@click.option("-e", "--end", type=float)
@click.option("-d", "--description", type=str, required=True)
@click.pass_context
def mark(
    ctx: click.Context,
    path: str,
    start: float,
    end: t.Optional[float],
    description: str,
) -> None:
    cli: CLI = ctx.obj["cli"]
    cli.mark(path, description, start, end)


@cli.command()
@click.argument("path", type=click.Path())
@click.pass_context
def show_references(ctx: click.Context, path: str) -> None:
    """Remove found references in a video file"""
    cli: CLI = ctx.obj["cli"]
    cli.show_references(path)


@cli.command()
@click.argument("path", type=click.Path())
@click.option("-o", "--output", type=click.Path(), default="out.mp4")
@click.pass_context
def shave(ctx: click.Context, path: str, output: str) -> None:
    """Remove found markings in a video file"""
    cli: CLI = ctx.obj["cli"]
    cli.shave(path, output)


if __name__ == "__main__":
    cli()
