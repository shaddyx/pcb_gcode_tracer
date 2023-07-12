import numpy as np


class GcodeGen:
    def __init__(self):
        pass

    def gen(self, lines):
        for x, y, x1, y1 in lines:
            yield self.move(x, y)
            yield self.down()
            yield self.move(x1, y1)
            yield self.up()

    def move(self, x, y):
        return f"G0 X{x}, Y{y}"

    def up(self):
        return f"G1 Z2"

    def down(self):
        return f"G1 Z0.3"