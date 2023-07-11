import numpy as np

import tracer.tracer_finders


def test_fund_next_clockwise():
    dots = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]
    ])
    assert tracer.tracer_finders.fund_next_clockwise(dots, 4, 1) == (5, 2)
    assert tracer.tracer_finders.fund_next_clockwise(dots, 5, 2) == (4, 3)
    assert tracer.tracer_finders.fund_next_clockwise(dots, 3, 4) == (3, 3)
    assert tracer.tracer_finders.fund_next_clockwise(dots, 1, 1) is None


def test_find_start():
    arr = np.zeros((3, 5), dtype=np.int32)
    assert tracer.tracer_finders.find_start(arr) is None
    arr[2, 4] = 1
    assert tracer.tracer_finders.find_start(arr) == (4, 2)
    arr[1, 2] = 1
    assert tracer.tracer_finders.find_start(arr) == (2, 1)
