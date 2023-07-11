import numba
import numpy as np

import tracer.tracer_finders
from tracer import tracer_tools


# @numba.jit(nopython=True)
def trace(im: np.array):
    h, w = im.shape
    while True:
        start = tracer.tracer_finders.find_start(im)
        if start is None:
            return
        startX, startY = start
        print(startX, startY)
        im[startY, startX] = 0
        yield start
