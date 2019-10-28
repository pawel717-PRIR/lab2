import timeit

from prir.config import app_logging
from prir.inspector.globals_inspector import get_caller_globals


def call_and_measure(code_to_call, number=100):
    elapsed_time, return_value = timeit.timeit(code_to_call, number=number, globals=get_caller_globals())
    # we are interested in mean elapsed time of single code invocation
    # so need to divide total time by number of invocations
    elapsed_time = elapsed_time / number
    logger.info('Elapsed time: {0:.5f}, Executed code: {1}'.format(elapsed_time, code_to_call))
    return return_value


timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        ret_val = {stmt}
    _t1 = _timer()
    return _t1 - _t0, ret_val
"""

logger = app_logging.get_logger('metrics')
