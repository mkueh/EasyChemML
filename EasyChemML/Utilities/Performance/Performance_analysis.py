"""
copy from https://github.com/XiekangZhang/MA_SRLB/blob/master/performance_analysis.py

thx XiekangZhang for the nice idea
"""

import tracemalloc
import functools
import time, os
from pympler import muppy, summary

def trace_memoryUsage(func):
    @functools.wraps(func)
    def wrapper_trace_memoryUsage(*args, **kwargs):
        # usage monitoring
        tracemalloc.start()

        value = func(*args, **kwargs)

        _, peak = tracemalloc.get_traced_memory()

        print(f'Process:{os.getpid()} Finished {func.__name__!r} with a memory peak of {peak / 10 ** 6:.2f}MB')
        tracemalloc.stop()
        return value

    return wrapper_trace_memoryUsage

def benchmark_function(func):
    @functools.wraps(func)
    def wrapper_benchmark_function(*args, **kwargs):
        # usage monitoring
        start_time = time.perf_counter()

        value = func(*args, **kwargs)

        end_time = time.perf_counter()
        run_time = end_time - start_time

        print(f'Process:{os.getpid()} Finished {func.__name__!r} in {run_time:.2f}s')
        return value

    return wrapper_benchmark_function
"""
def printMemoryFootprint():
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)  # Prints out a summary of the large objects
    summary.print_(sum1)
"""

"""
tracemalloc.start()
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("\n [ Top 10 ] - Memoryusage")
for stat in top_stats[:10]:
    print(stat)

tracemalloc.stop()
"""