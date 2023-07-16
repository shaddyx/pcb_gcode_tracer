import numba
import numpy as np

from tracer import tracer_tools, tracer_finders, tracer_constants, tracer_gen
from tracer.tracer_jit import tjit



@tjit(nopython=True)
def trace(im: np.array):
    while True:
        start = tracer_finders.find_start(im)
        if start == tracer_constants.XY_NOT_FOUND:
            return

        while True:
            start_x, start_y = start
            line = tracer_finders.find_next_line(im, start_x, start_y)
            for kx, ky in tracer_gen.line_gen(line[0], line[1], line[2], line[3]):
                im[ky, kx] = 0
            yield line
            
            start = tracer_finders.find_next_dot_clockwise(im, line[2], line[3])
            if start == tracer_constants.XY_NOT_FOUND:
                break
