import logging

import PIL
import numpy as np

from gcode import gcode_gen
from tracer import tracer_main

logging.basicConfig(level=logging.INFO)
import cairosvg
from PIL import Image

import config
import image_util
logging.info("Converting to image")
cairosvg.svg2png(
    url=config.INPUT_FILE,
    write_to=config.TMP_IMG,
    dpi=config.DPI * config.MULTIPLICATION_RATIO
    # output_width=1000,
    # output_height=1000
)


def save_traced(traced_data):
    ggen = gcode_gen.GcodeGen()
    with open("output.gcode", "w") as f:
        for k in ggen.gen(traced_data):
            f.write(k + "\n")


with Image.open(config.TMP_IMG) as im:
    px = im.load()
    npx = np.asarray(im)
    logging.info(f"Searching min/max for {npx.shape}")
    mx, my, max_x, max_y = image_util.find_min_max(npx)
    logging.info("cropping")
    im1 = npx[my:max_y, mx:max_x]
    logging.info("Converting to bw")
    bw = image_util.convert_to_bw(im1)
    traced_data = tracer_main.trace(image_util.resample(image_util.convert_to_one_bit(bw), 5))
    save_traced(traced_data)
    im1 = PIL.Image.fromarray(bw)
    im1.save(config.TMP_BW_IMG)
    #npx = np.array(im1)

    # print(npx)
    #
