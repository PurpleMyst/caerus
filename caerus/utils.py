import os
import json
import sqlite3
import typing as t
from contextlib import contextmanager
from itertools import repeat
import subprocess

import cv2
import numpy as np
import numpy.typing as npt

Frame = npt.NDArray[np.uint8]
PathArg = t.Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]


@contextmanager
def releasing(cap: cv2.VideoCapture) -> t.Iterator[cv2.VideoCapture]:
    try:
        yield cap
    finally:
        cap.release()


def insert_if_not_exists(
    db: sqlite3.Connection,
    table: str,
    values: t.Dict[str, t.Any],
) -> int:
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


class FFMpeg:
    def __init__(self, **options: t.Any) -> None:
        self.options = options

    def __call__(
        self, *args: t.Any, **kwargs: t.Any
    ) -> subprocess.CompletedProcess[str]:
        argv = ["ffmpeg"]
        for k, v in self.options.items():
            argv.append("-" + k)
            argv.append(str(v))
        argv.extend(map(str, args))
        return subprocess.run(argv, **kwargs)


def find_series(path: PathArg) -> str:
    info = json.loads(
        subprocess.run(
            ("filebot", "-mediainfo", "--format", "{json}", path),
            check=True,
            capture_output=True,
        ).stdout.decode("ascii", "ignore")
    )
    series: str = info["seriesInfo"]["name"]
    return series
