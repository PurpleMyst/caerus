import json
import sqlite3
import subprocess
import typing as t
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from inspect import getcallargs
from itertools import repeat

import cv2
import numpy as np
import numpy.typing as npt
import structlog

Frame = npt.NDArray[np.uint8]
F = t.TypeVar("F", bound=t.Callable[..., t.Any])


def format_time(s: float, *, sep: str = ":") -> str:
    h, m = divmod(s, 60)
    return f"{h:02.0f}{sep}{m:06.3f}"


def log_parameters(*, ignore: t.Sequence[str] = ()) -> t.Callable[[F], F]:
    def decorator(func: F) -> F:
        name = func.__name__

        @wraps(func)
        def inner(*args: t.Any, **kwargs: t.Any) -> t.Any:
            logger = structlog.get_logger()
            try:
                callargs = getcallargs(func, *args, **kwargs)
            except Exception as error:
                logger.warning("callargs failed", func=name, error=error)
            else:
                for arg in ignore:
                    del callargs[arg]
                logger.debug("log_parameters.call", func=name, **callargs)
            return func(*args, **kwargs)

        return t.cast(F, inner)

    return decorator


@contextmanager
def releasing(cap: cv2.VideoCapture) -> t.Iterator[cv2.VideoCapture]:
    try:
        yield cap
    finally:
        cap.release()


def insert_unique(db: sqlite3.Connection, table: str, **values: t.Any) -> int:
    """Insert a row into a database table making sure it is unique, returning its id.

    Uniqueness is determined by the first element of VALUES, in insertion order."""
    uniq_column, uniq_value = next(iter(values.items()))
    result = db.execute(
        f"SELECT id FROM {table} WHERE {uniq_column} = ?", (uniq_value,)
    ).fetchone()
    id: int
    if result is None:
        id = db.execute(
            f"INSERT INTO {table}({','.join(values.keys())}) "
            f"VALUES ({','.join(repeat('?', len(values)))})",
            tuple(values.values()),
        ).lastrowid
    else:
        [id] = result
    return id


@dataclass
class FFMpeg:
    options: dict[str, t.Any] = field(default_factory=dict)

    def __call__(
        self, *args: t.Any, **kwargs: t.Any
    ) -> subprocess.CompletedProcess[str]:
        kwargs.setdefault("check", True)
        cmd = ["ffmpeg"]
        *head, tail = args
        cmd.extend(map(str, head))
        for k, v in self.options.items():
            cmd.append("-" + k)
            cmd.append(str(v))
        cmd.append(str(tail))
        return subprocess.run(cmd, **kwargs)


def find_series(path: str) -> str:
    info = json.loads(
        subprocess.run(
            ("filebot", "-mediainfo", "--format", "{json}", path),
            check=True,
            capture_output=True,
        ).stdout.decode("ascii", "ignore")
    )
    series: str = info["seriesInfo"]["name"]
    return series
