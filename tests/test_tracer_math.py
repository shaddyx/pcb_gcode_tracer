from tracer import tracer_math


def test_line_len():
    assert tracer_math.line_len(0, 0, 100, 100) == 141.4213562373095


def test_line_len_tuples():
    assert tracer_math.line_len_tuples((0, 0), (100, 100)) == 141.4213562373095


def test_get_opposite_dot():
    assert tracer_math.get_opposite_dot(0, 0, 1, 1) == (2, 2)
    assert tracer_math.get_opposite_dot(1, 0, 1, 1) == (1, 2)
    assert tracer_math.get_opposite_dot(5, 5, 4, 4) == (3, 3)
