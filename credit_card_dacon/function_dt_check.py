import time
from functools import wraps


def time_checker(func):
    @wraps(func) 
    # 데코레이터를 썼을때 독스트링 등의 함수 고유 정보가 사라지는 것을 방지
    def inner_function(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("{} learning time: {}".format(func.__name__, end_time - start_time))
        return result
    return inner_function