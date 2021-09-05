import typing as t
import sqlite3
import os
from itertools import repeat
from contextlib import contextmanager

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
