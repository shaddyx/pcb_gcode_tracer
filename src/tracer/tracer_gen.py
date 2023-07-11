import numba


@numba.jit(nopython=True)
def line_gen(x: int, y: int, x1: int, y1: int):
    dx = abs(x1 - x)
    dy = abs(y1 - y)
    sx = -1 if x > x1 else 1
    sy = -1 if y > y1 else 1
    err = dx - dy

    while True:
        yield x, y

        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
