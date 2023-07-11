import numba
import numpy as np
import color_util
from tracer import tracer_math


@numba.jit(nopython=True)
def get_dot_safe(im: np.array, x: int, y: int):
    if x < 0 or y < 0 or x >= im.shape[1] or y >= im.shape[0]:
        return None
    return im[y, x]


@numba.jit(nopython=True)
def check_dot_safe(im: np.array, x: int, y: int, val: int):
    return get_dot_safe(im, x, y) == val


@numba.jit(nopython=True)
def is_bounding_dot(im: np.array, x: int, y: int):
    h, w = im.shape

    return check_dot_safe(im, x, y, 1) and (
            x == w - 1 or y == h - 1 or x == 0 or y == 0
            or check_dot_safe(im, x - 1, y, 0)
            or check_dot_safe(im, x, y - 1, 0)
            or check_dot_safe(im, x + 1, y, 0)
            or check_dot_safe(im, x, y + 1, 0)
    )


@numba.jit(nopython=True)
def is_bounding_line(im: np.array, x, y, x1, y1):
    for xx, yy in tracer_math.line_gen(x, y, x1, y1):
        if not is_bounding_dot(im, xx, yy):
            return False
    return True


@numba.jit(nopython=True)
def find_start(im: np.array):
    h, w = im.shape
    for y in range(h):
        for x in range(w):
            if im[y, x] == 1:
                return x, y
    return None


@numba.jit(nopython=True)
def convert_to_onebit(im: np.array):
    h, w, bts = im.shape
    out = np.zeros((h, w), dtype=np.int8)
    for y in range(h):
        for x in range(w):
            out[y, x] = 0 if color_util.is_void(im[y, x]) else 1
    return out
