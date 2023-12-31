import numpy

from tracer.tracer_jit import tjit


@tjit(nopython=True)
def is_black(px):
    if px[0] == 255 and px[1] == 255 and px[2] == 255:
        return False
    if px[3] > 120:
        return True
    # if (px[0] > 250 or px[1] > 250 or px[2]):
    #     return True
    return False


@tjit(nopython=True)
def is_void(px: numpy.array):
    return px[0] == 0 and px[1] == 0 and px[2] == 0 and px[2] == 0
