import numpy as np

from tracer import tracer_tools


def test_get_dot_safe():
    arr = np.zeros((5, 3), dtype=np.int32)
    assert tracer_tools.get_dot_safe(arr, 1, 1) == 0
    assert tracer_tools.get_dot_safe(arr, 0, 0) == 0
    assert tracer_tools.get_dot_safe(arr, -1, 0) is None
    assert tracer_tools.get_dot_safe(arr, 2, 4) == 0
    assert tracer_tools.get_dot_safe(arr, 3, 0) is None


def test_is_bounding_dot():
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
    ]
    )
    assert tracer_tools.is_bounding_dot(dots, 3, 2)
    assert not tracer_tools.is_bounding_dot(dots, 4, 2)
    assert tracer_tools.is_bounding_dot(dots, 5, 2)
    assert tracer_tools.is_bounding_dot(dots, 6, 8)
    assert not tracer_tools.is_bounding_dot(dots, 0, 0)
    assert not tracer_tools.is_bounding_dot(dots, 1, 1)


def test_find_start():
    arr = np.zeros((3, 5), dtype=np.int32)
    assert tracer_tools.find_start(arr) is None
    arr[2, 4] = 1
    assert tracer_tools.find_start(arr) == (4, 2)
    arr[1, 2] = 1
    assert tracer_tools.find_start(arr) == (2, 1)


def test_convert_to_onebit():
    arr = np.zeros((3, 5, 4), dtype=np.int32)
    res = tracer_tools.convert_to_onebit(arr)
    for x in np.nditer(res):
        assert x == 0


def test_convert_to_onebit_ones():
    arr = np.zeros((3, 5, 4), dtype=np.int32)
    arr[:] = (255, 255, 255, 255)
    res = tracer_tools.convert_to_onebit(arr)
    for x in np.nditer(res):
        assert x == 1


def test_is_bounding_line():
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
    assert tracer_tools.is_bounding_line(dots, 3, 2, 3, 6)
    assert not tracer_tools.is_bounding_line(dots, 3, 2, 3, 7)
    assert tracer_tools.is_bounding_line(dots, 2, 5, 5, 2)


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
    assert tracer_tools.fund_next_clockwise(dots, 4, 1) == (5, 2)
    assert tracer_tools.fund_next_clockwise(dots, 5, 2) == (4, 3)
    assert tracer_tools.fund_next_clockwise(dots, 3, 4) == (3, 3)
    assert tracer_tools.fund_next_clockwise(dots, 1, 1) is None

