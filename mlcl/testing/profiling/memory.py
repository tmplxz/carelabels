import os
import threading

import numpy as np
import psutil

from .profiling import Profiling
from mlcl.util import encode


class MeasureMemory(threading.Thread):
    def run(self):
        self.running = True

        currentProcess = psutil.Process(os.getpid())
        self.res = []
        while self.running:
            self.res.append(currentProcess.memory_full_info().rss)

    def stop(self):
        self.running = False


class MemoryProfiling(Profiling):

    def run(self, func):
        p = psutil.Process(os.getpid())
        display_memory = MeasureMemory()
        mem_results = {}
        display_memory.start()
        mem_results['PRE'] = p.memory_full_info().rss
        result = func()
        display_memory.stop()
        display_memory.join()
        mem_results['ENC'] = encode(display_memory.res)
        return mem_results, result
