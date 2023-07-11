import numba
import numpy as np

from tracer.tracer_tools import check_black_dot_safe


@numba.jit(nopython=True)
def fund_next_clockwise(im: np.array, x: int, y: int):
    if check_black_dot_safe(im, x, y - 1):
        return x, y - 1
    if check_black_dot_safe(im, x + 1, y - 1):
        return x + 1, y - 1
    if check_black_dot_safe(im, x + 1, y):
        return x + 1, y
    if check_black_dot_safe(im, x + 1, y + 1):
        return x + 1, y + 1
    if check_black_dot_safe(im, x, y + 1):
        return x, y + 1
    if check_black_dot_safe(im, x - 1, y + 1):
        return x - 1, y + 1
    if check_black_dot_safe(im, x - 1, y):
        return x - 1, y
    if check_black_dot_safe(im, x - 1, y - 1):
        return x - 1, y - 1
    return None


@numba.jit(nopython=True)
def find_start(im: np.array):
    h, w = im.shape
    for y in range(h):
        for x in range(w):
            if im[y, x] == 1:
                return x, y
    return None
