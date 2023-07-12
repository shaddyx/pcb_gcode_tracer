import numba
import numpy as np
from PIL import Image

import color_util
import config
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


def resample(img: Image, multiplication_ration: float) -> Image:
    output_im = Image.new('RGBA', (
        int(img.width / multiplication_ration) + 1, int(img.height / multiplication_ration) + 1))

    for x in range(output_im.width):
        for y in range(output_im.height):
            source_coords = (x * config.MULTIPLICATION_RATIO, y * config.MULTIPLICATION_RATIO)
            pix = img.getpixel(source_coords)
            if color_util.is_black(pix):
                output_im.putpixel((x, y), (1, 0, 0, 255))
            else:
                output_im.putpixel((x, y), (0, 0, 0, 0))
    return output_im
