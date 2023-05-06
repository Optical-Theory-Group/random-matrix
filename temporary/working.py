import time

import numpy as np

from random_matrix.utils.memoize import Memoize, memoize

@memoize(cache_filename="outer.json")
def outer_func(x):
    time.sleep(np.random.uniform(1,5))
    return 2*inner_func(x)

@memoize(cache_filename="inner.json")
def inner_func(x):
    time.sleep(np.random.uniform(1,5))
    return 2*x


for i in range(5):
    print(i)
    outer_func(i)