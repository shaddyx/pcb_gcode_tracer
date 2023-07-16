import numpy as np

from tracer import tracer_tools, tracer_constants, tracer_math
from tracer.tracer_jit import tjit


@tjit(nopython=True)
def find_start_from_beginning(im: np.array):
    h, w = im.shape
    for y in range(h):
        for x in range(w):
            if tracer_tools.is_bounding_dot(im, x, y):
                return x, y
    return tracer_constants.XY_NOT_FOUND


@tjit(nopython=True)
def find_in_sector(im: np.array, x: int, y: int, x1: int, y1: int):
    min_res = tracer_constants.XY_NOT_FOUND
    min_range = 9999999999999999
    if x != x1:
        ix = int(abs(x1 - x) / (x1 - x))
    else:
        ix = 1
    if y != y1:
        iy = int(abs(y1 - y) / (y1 - y))
    else:
        iy = 1

    for cx in range(x, x1, ix):
        for cy in range(y, y1, iy):
            if tracer_tools.is_bounding_dot(im, cx, cy):
                length = tracer_math.line_len(x, y, cx, cy)
                if min_range > length:
                    min_res = (cx, cy)
                    min_range = length
    return min_res


@tjit(nopython=True)
def find_start(im: np.array, x: int = tracer_constants.NO_VALUE_DOT, y: int = tracer_constants.NO_VALUE_DOT):
    if x == tracer_constants.NO_VALUE_DOT or y == tracer_constants.NO_VALUE_DOT:
        return find_start_from_beginning(im)
    h, w = im.shape
    res = [find_in_sector(im, x, y, 0, 0),
           find_in_sector(im, x, y, w, 0),
           find_in_sector(im, x, y, w, h),
           find_in_sector(im, x, y, 0, h)]
    res = [k for k in res if k != tracer_constants.XY_NOT_FOUND]
    ranges = [tracer_math.line_len_tuples((x, y), k) for k in res]
    min_res = tracer_constants.XY_NOT_FOUND
    min_range = 9999999999999999
    for k in range(len(ranges)):
        if min_range > ranges[k]:
            min_range = ranges[k]
            min_res = res[k]
    return min_res

