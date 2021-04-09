import time

import numpy as np

from .profiling import Profiling


class RuntimeProfiling(Profiling):

    def run(self, func):

        t1 = time.time()
        func()
        t2 = time.time()

        return {
            'WCT': t2 - t1,
        }
