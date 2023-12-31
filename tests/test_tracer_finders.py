import numpy as np

from tracer import tracer_finders, tracer_constants


def test_find_next_clockwise():
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
    assert tracer_finders.find_next_dot_clockwise(dots, 4, 1) == (5, 2)
    assert tracer_finders.find_next_dot_clockwise(dots, 5, 2) == (4, 3)
    assert tracer_finders.find_next_dot_clockwise(dots, 3, 4) == (3, 3)
    assert tracer_finders.find_next_dot_clockwise(dots, 1, 1) == tracer_constants.XY_NOT_FOUND
    assert tracer_finders.find_next_dot_clockwise(dots, 2, 5, 3, 4) == (3, 5)
    # detect opposite
    assert tracer_finders.find_next_dot_clockwise(dots, 3, 3, 3, 4) == (3, 2)
    assert tracer_finders.find_next_dot_clockwise(dots, 3, 3, 3, 2) == (3, 4)


def test_find_start():
    arr = np.zeros((3, 5), dtype=np.int32)
    assert tracer_finders.find_start(arr) == tracer_constants.XY_NOT_FOUND
    arr[2, 4] = 1
    assert tracer_finders.find_start(arr) == (4, 2)
    arr[1, 2] = 1
    assert tracer_finders.find_start(arr) == (2, 1)


def test_find_max_line():
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

    assert tracer_finders.find_max_line(dots, 3, 2, 3, 6) == (3, 2, 3, 6)
    assert tracer_finders.find_max_line(dots, 3, 2, 3, 9) == (3, 2, 3, 6)
    assert tracer_finders.find_max_line(dots, 2, 5, 5, 2) == (2, 5, 5, 2)


def test_find_next_line():
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
    assert tracer_finders.find_next_line(dots, 3, 6) == (3, 6, 3, 2)
    assert tracer_finders.find_next_line(dots, 3, 2) == (3, 2, 3, 6)
    dots[1, 4] = 0
    assert tracer_finders.find_next_line(dots, 3, 2) == (3, 2, 5, 2)
