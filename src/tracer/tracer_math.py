import math

import numba

from tracer.tracer_jit import tjit


@tjit(nopython=True)
def line_len_tuples(dot_a, dot_b):
    return line_len(dot_a[0], dot_a[1], dot_b[0], dot_b[1])


@tjit(nopython=True)
def line_len(x: int, y: int, x1: int, y1: int):
    return math.sqrt(pow(x - x1, 2) + pow(y - y1, 2))


@tjit(nopython=True)
def neibours_dots(x: int, y: int, x1: int, y1: int):
    return abs(x1 - x) <= 1 and abs(y1 - y) <= 1


@tjit(nopython=True)
def get_opposite_dot(x: int, y: int, x1: int, y1: int):
    xx = x1 - x
    yy = y1 - y
    return x1 + xx, y1 + yy