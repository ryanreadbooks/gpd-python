import time
import functools
from typing import Callable


def timer(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        st = time.time()
        ret = func(*args, **kwargs)
        et = time.time()
        print(f'{func.__name__} took time {et - st} s.')
        return ret
    return wrapper
