from tracer import tracer_math


def test_line_len():
    assert tracer_math.line_len(0, 0, 100, 100) == 141.4213562373095


def test_line_len_tuples():
    assert tracer_math.line_len_tuples((0, 0), (100, 100)) == 141.4213562373095
