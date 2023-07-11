import math

import numba


@numba.jit(nopython=True)
def line_len_tuples(dot_a, dot_b):
    return line_len(dot_a[0], dot_a[1], dot_b[0], dot_b[1])


@numba.jit(nopython=True)
def line_len(x: int, y: int, x1: int, y1: int):
    return math.sqrt(pow(x - x1, 2) + pow(y - y1, 2))
