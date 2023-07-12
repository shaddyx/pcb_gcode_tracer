import numba
import numpy as np

from tracer import tracer_tools, tracer_gen


@numba.jit(nopython=True)
def find_next_dot_clockwise(im: np.array, x: int, y: int, ox: int | None = None, oy: int | None = None):
    _clockwise = [
        (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)
    ]
    for k in _clockwise:
        xx, yy = x + k[0], y + k[1]
        if ox == xx and oy == yy:
            continue
        if tracer_tools.is_bounding_dot(im, xx, yy):
            return xx, yy
    return -1, -1


@numba.jit(nopython=True)
def find_start(im: np.array):
    h, w = im.shape
    for y in range(h):
        for x in range(w):
            if im[y, x] == 1:
                return x, y
    return None


@numba.jit(nopython=True)
def find_max_line(im: np.array, x: int, y: int, x1: int, y1: int):
    px, py = x, y
    for dx, dy in tracer_gen.line_gen(x, y, x1, y1):
        if not tracer_tools.is_bounding_dot(im, dx, dy):
            return x, y, px, py
        px, py = dx, dy
    return x, y, x1, y1


@numba.jit(nopython=True)
def _check_prev(x: int, y: int, nx: int, ny: int):
    first = True
    for ppx, ppy in tracer_gen.line_gen(nx, ny, x, y):
        if first:
            first = False
            continue
        return int(ppx), int(ppy)
    return nx, ny


@numba.jit(nopython=True)
def find_next_line(im: np.array, x: int, y: int):
    px, py = x, y
    xx, yy = x, y
    while True:
        nd = find_next_dot_clockwise(im, xx, yy, px, py)
        if nd == (-1, -1):
            return x, y, xx, yy
        ndx, ndy = nd[0], nd[1]
        cx, cy = _check_prev(x, y, ndx, ndy)
        if cx != xx or cy != yy:
            return x, y, xx, yy
        if tracer_tools.is_bounding_line(im, x, y, ndx, ndy):
            px, py = xx, yy
            xx, yy = nd
        else:
            return x, y, xx, yy
