import numba


def tjit(*args, **kwargs):
    def decorator(fn):
        return fn

    #return numba.jit(*args, **kwargs)
    return decorator
