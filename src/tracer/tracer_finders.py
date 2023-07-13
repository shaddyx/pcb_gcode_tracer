import numba
import numpy as np

from tracer import tracer_tools, tracer_gen, tracer_constants, tracer_math
from tracer.tracer_jit import tjit


@tjit(nopython=True)
def find_next_dot_clockwise(im: np.array, x: int, y: int, ox: int = tracer_constants.NO_VALUE_DOT, oy: int = tracer_constants.NO_VALUE_DOT):
    if ox != tracer_constants.NO_VALUE_DOT and (x != ox or y != oy):
        xx, yy = tracer_math.get_opposite_dot(ox, oy, x, y)
        if tracer_tools.is_bounding_dot(im, xx, yy):
            return xx, yy

    _clockwise = [
        (0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1)
    ]
    for k in _clockwise:
        xx, yy = x + k[0], y + k[1]
        if ox == xx and oy == yy:
            continue
        if tracer_tools.is_bounding_dot(im, xx, yy):
            return xx, yy
    return tracer_constants.XY_NOT_FOUND


@tjit(nopython=True)
def find_start(im: np.array):
    h, w = im.shape
    for y in range(h):
        for x in range(w):
            if im[y, x] == 1:
                return x, y
    return tracer_constants.XY_NOT_FOUND


@tjit(nopython=True)
def find_max_line(im: np.array, x: int, y: int, x1: int, y1: int):
    px, py = x, y
    for dx, dy in tracer_gen.line_gen(x, y, x1, y1):
        if not tracer_tools.is_bounding_dot(im, dx, dy):
            return x, y, px, py
        px, py = dx, dy
    return x, y, x1, y1


@tjit(nopython=True)
def _check_prev(x: int, y: int, nx: int, ny: int):
    first = True
    for ppx, ppy in tracer_gen.line_gen(nx, ny, x, y):
        if first:
            first = False
            continue
        return int(ppx), int(ppy)
    return nx, ny


@tjit(nopython=True)
def find_next_line(im: np.array, x: int, y: int):
    px, py = x, y
    xx, yy = x, y
    while True:
        nd = find_next_dot_clockwise(im, xx, yy, px, py)
        if nd == tracer_constants.XY_NOT_FOUND:
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
