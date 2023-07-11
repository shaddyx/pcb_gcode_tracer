from tracer import tracer_math


def test_line_gen():
    res = list(tracer_math.line_gen(10, 10, 15, 15))
    assert res == [(10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15)]

    res = list(tracer_math.line_gen(10, 10, 10, 15))
    assert res == [(10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15)]

    res = list(tracer_math.line_gen(10, 10, 10, 5))
    assert res == [(10, 10), (10, 9), (10, 8), (10, 7), (10, 6), (10, 5)]
