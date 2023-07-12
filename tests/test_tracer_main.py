import numpy as np

from tracer import tracer_main


def test_trace():
    dots = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    )

    assert list(tracer_main.trace(dots)) == [(3, 2, 4, 2), (3, 3, 4, 3), (3, 4, 3, 6), (2, 5, 2, 6)]


def test_trace():
    dots = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]
    )

    assert list(tracer_main.trace(dots)) == [(0, 0, 7, 0), (0, 1, 0, 7), (7, 1, 7, 8), (2, 2, 4, 4), (5, 3, 5, 3),
                                             (3, 5, 2, 6), (5, 5, 5, 5), (0, 8, 6, 8)]
