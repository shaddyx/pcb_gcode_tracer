import numpy as np

from tracer import tracer_constants, tracer_math


class GcodeGen:
    def __init__(self):
        self.prev = tracer_constants.XY_NOT_FOUND
        self.dpi = 96
        self.dpmm = self.dpi / 25.4
        self.scale = 1
        self.is_up = True
        self.distance_threshold = 0.1

    def convert_to_mm(self, value):
        return round(value / self.scale * 1000 / self.dpmm) / 1000

    def down_if_required(self):
        if self.is_up:
            self.is_up = False
            yield self.down()

    def up_if_required(self, x, y):
        if not self.is_up and (
                self.prev != tracer_constants.XY_NOT_FOUND
                and not tracer_math.neibours_dots(self.prev[0], self.prev[1], x, y)
        ):
            self.is_up = True
            yield self.up()

    def gen(self, lines):
        self.prev = tracer_constants.XY_NOT_FOUND
        self.is_up = True

        for x, y, x1, y1 in lines:
            xx, yy, xx1, yy1 = self.convert_to_mm(x), self.convert_to_mm(y), self.convert_to_mm(x1), self.convert_to_mm(
                y1)
            yield from self.up_if_required(x, y)
            yield self.move(xx, yy)
            yield from self.down_if_required()
            yield self.move(xx1, yy1)
            self.prev = (x1, y1)

    def move(self, x, y):
        return f"G0 X{x}, Y{y}"

    def up(self):
        return f"G1 Z2"

    def down(self):
        return f"G1 Z0.3"
