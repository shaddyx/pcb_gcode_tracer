import numba
import numpy as np

from tracer import tracer_tools


@numba.jit(nopython=True)
def find_next_clockwise(im: np.array, x: int, y: int, ox: int | None = None, oy: int | None = None):
    _clockwise = [
        (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)
    ]
    for k in _clockwise:
        xx, yy = x + k[0], y + k[1]
        if ox == xx and oy == yy:
            continue
        if tracer_tools.is_bounding_dot(im, xx, yy):
            return xx, yy
    return None


@numba.jit(nopython=True)
def find_start(im: np.array):
    h, w = im.shape
    for y in range(h):
        for x in range(w):
            if im[y, x] == 1:
                return x, y
    return None


def find_next_line(im: np.array, x: int, y: int):
    prev = None
    while True:
        next = find_next_clockwise(im, x, y)
        if next is None:
            pass
