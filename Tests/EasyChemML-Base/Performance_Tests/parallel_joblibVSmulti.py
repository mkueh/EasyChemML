"""
This tests issue https://github.com/scikit-learn/scikit-learn/issues/3835
"""

import time
import itertools
import joblib
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm as progress_bar


class Timer(object):

    def __init__(self, title=""):
        self.title = title

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.interval = time.time() - self.start
        print("{}: time elapsed: {}".format(self.title, self.interval))


def delayed(func):
    def wrapper(*args, **kwargs):
        return partial(func, *args, **kwargs)
    return wrapper


def call(f):
    return f()


def Parallel(n_jobs=1, chunksize=12, ordered=True, **kwargs):
    def run(args):
        if n_jobs == 1:
            # sequential mode (useful for debugging)
            yield from map(call, args)
        else:
            # spawn workers
            pool = Pool(n_jobs, **kwargs)
            try:
                if ordered:
                    yield from pool.imap(call, args, chunksize=chunksize)
                else:
                    yield from pool.imap_unordered(call, args, chunksize=chunksize)
            except KeyboardInterrupt:
                pool.terminate()
            except Exception:
                pool.terminate()
                raise
            else:
                pool.close()
            finally:
                pool.join()
    return run


def factorial(num, the_dict):
    result = 1
    for i in range(1, num):
        result *= i
    return result


def multi_factorial(num_range, the_dict):
    return [factorial(num, the_dict) for num in num_range]


def run_parallel(func, huge_dict, n_jobs=4, backend="joblib", show_progress=True):
    args = [(range(i * 25, (i + 1) * 25), huge_dict) for i in range(n_jobs)]
    if show_progress:
        args = progress_bar(args, desc=f"{n_jobs} workers")
    if backend == "joblib":
        return joblib.Parallel(n_jobs=n_jobs, batch_size=12)(joblib.delayed(func)(*arg) for arg in args)
    elif backend == "multiproc":
        return Parallel(n_jobs=n_jobs)(delayed(func)(*arg) for arg in args)
    else:
        raise ValueError("invalid backend")


def build_data():
    huge_dict = {}
    for string in itertools.combinations('abcdefghijklmnopqrstuvwxyz', 11):
        if 'a' in string:
            continue
        huge_dict[string] = 'gaopjfw9ezucr209jf82urpioewsackjrp'
    return huge_dict


def main():
    huge_dict = build_data()

    with Timer(title="multiproc"):
        result = run_parallel(multi_factorial, huge_dict, n_jobs=12, backend="multiproc")
        result1 = list(itertools.chain(*result))

    with Timer(title="joblib"):
        result = run_parallel(multi_factorial, huge_dict, n_jobs=12, backend="joblib")
        result2 = list(itertools.chain(*result))

    assert result1 == result2


if __name__ == '__main__':
    main()