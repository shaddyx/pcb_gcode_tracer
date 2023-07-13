import numpy as np
from PIL import Image

import color_util
from tracer.tracer_jit import tjit


@tjit(nopython=True)
def find_min_max(im: np.array):
    h, w, bts = im.shape
    mx = w
    my = h
    max_x = 0
    max_y = 0
    for x in range(0, w):
        for y in range(0, h):
            if color_util.is_black(im[y, x]):
                mx = min(x, mx)
                my = min(y, my)
                max_x = max(x, max_x)
                max_y = max(y, max_y)
    return mx, my, max_x, max_y


@tjit(nopython=True)
def convert_to_bw(img: Image) -> Image:
    h, w, bts = img.shape
    output_im = img.copy()
    for x in range(w):
        for y in range(h):
            pix = img[y, x]
            if color_util.is_black(pix):
                output_im[y, x] = (1, 0, 0, 255)
            else:
                output_im[y, x] = (0, 0, 0, 0)
    return output_im

@tjit(nopython=True)
def convert_to_one_bit(im: np.array) -> np.array:
    h, w, bts = im.shape
    output_im = np.zeros((h, w))
    for x in range(w):
        for y in range(h):
            if color_util.is_black(im[y, x]):
                output_im[y, x] = 1
            else:
                output_im[y, x] = 0
    return output_im


@tjit(nopython=True)
def resample(img: np.array, multiplication_ratio: float) -> Image:
    h, w = img.shape
    oh, ow = int(h / multiplication_ratio), int(w / multiplication_ratio)
    print(f"resampling {w}*{h} to {ow}*{oh}")
    output_im = np.zeros((oh, ow))

    for x in range(ow):
        for y in range(oh):
            pix = img[int(y * multiplication_ratio), int(x * multiplication_ratio)]
            if pix:
                output_im[y, x] = 1
            else:
                output_im[y, x] = 0
    return output_im
