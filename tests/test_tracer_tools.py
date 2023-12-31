import numpy as np

from tracer import tracer_tools, tracer_constants


def test_get_dot_safe():
    arr = np.zeros((5, 3), dtype=np.int32)
    assert tracer_tools.get_dot_safe(arr, 1, 1) == 0
    assert tracer_tools.get_dot_safe(arr, 0, 0) == 0
    assert tracer_tools.get_dot_safe(arr, -1, 0) == tracer_constants.NO_VALUE_DOT
    assert tracer_tools.get_dot_safe(arr, 2, 4) == 0
    assert tracer_tools.get_dot_safe(arr, 3, 0) == tracer_constants.NO_VALUE_DOT


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
