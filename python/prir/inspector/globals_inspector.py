import inspect


def get_caller_globals():
    return inspect.stack()[2][0].f_globals
