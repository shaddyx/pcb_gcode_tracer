import numpy as np

import config
from tracer import tracer_constants, tracer_math


class GcodeGen:
    def __init__(self):
        self.prev = tracer_constants.XY_NOT_FOUND
        self.dpi = 96
        self.dpmm = self.dpi / 25.4
        self.scale = 2
        self.is_up = True
        self.distance_threshold = 0.1
        self.xy = (0, 0)
        self.z = 0
        self.relative = config.RELATIVE

    def round(self, v):
        return round(v * 100) / 100

    def convert_to_mm(self, value):
        return self.round(value / self.scale / self.dpmm)

    def down_if_required(self):
        if self.is_up:
            self.is_up = False
            yield self.down()

    def to_relative(self, x, y):
        if not self.relative:
            return x, y
        res = (x - self.xy[0], y - self.xy[1])
        self.xy = (x, y)
        return res

    def to_relative_z(self, z):
        if not self.relative:
            return z
        res = z - self.z
        self.z = z
        return res

    def up_if_required(self, x, y):
        if not self.is_up and (
                self.prev != tracer_constants.XY_NOT_FOUND
                and not tracer_math.neibours_dots(self.prev[0], self.prev[1], x, y)
        ):
            self.is_up = True
            yield self.up()

    def gen(self, lines):
        yield from self.init()
        if self.relative:
            yield self.set_relative()
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
        yield self.up()
        if self.relative:
            yield self.set_absolute()
        yield self.go_home()

    def set_absolute(self):
        return "G90"

    def move(self, x, y):
        x, y = self.to_relative(x, y)
        x = self.round(x)
        y = self.round(y)
        t = "G0" if self.is_up else "G1"
        return f"{t} X{x}, Y{y}"

    def up(self):
        return f"G1 Z1"

    def down(self):
        return f"G1 Z-1"

    def set_relative(self):
        return f"G91"

    def go_home(self):
        return "G28"

    def init(self):
        yield "M201 X300 Y300"
        yield "M203 X10 Y10"
