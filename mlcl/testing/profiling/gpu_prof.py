from collections import defaultdict
from multiprocessing import Queue, Process
from datetime import datetime
import time

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlDeviceGetMemoryInfo
from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU

from mlcl.testing.profiling.profiling import Profiling


class GpuMonitoringProcess(Profiling):

    def run(self, func):
        queue = Queue()

        p = MonitoringProcess(lambda: get_gpu_stats_nvml_py(), queue)
        p.start()
        result = func()
        p.terminate()

        out = defaultdict(list)
        while not queue.empty():
            x = queue.get()
            for key, val in x.items():
                out[key].append(val)

        return out, result


class MonitoringProcess(Process):
    def __init__(self, measurement_fn, queue, interval=0.1, timeout=365 * 24 * 60 * 60 * 10):
        """
        This allows to monitor measurements via
        :param measurement_fn:
        :param queue:
        :param interval:
        :param timeout: seconds to run the process (Default is 10 years:D)
        """
        super(MonitoringProcess, self).__init__()

        self.measurement_fn = measurement_fn
        self.queue = queue
        self.interval = interval

        self.timeout = timeout

    def run(self):
        process_start = time.time()

        while time.time() < process_start + self.timeout:
            start = time.time()

            measurement = self.measurement_fn()
            self.queue.put(measurement)

            measure_duration = time.time() - start
            sleep_time = self.interval - measure_duration

            if sleep_time > 0:
                time.sleep(sleep_time)


def get_gpu_stats_nvml_py(gpu_id=0):
    # How to get device handle?
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_id)

    # Energy
    milliWatts = nvmlDeviceGetPowerUsage(handle)

    # Memory
    memory_t = nvmlDeviceGetMemoryInfo(handle)
    # Utilization
    utilization_t = nvmlDeviceGetUtilizationRates(handle)

    tmp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

    unix_time_millis = (datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0

    return {
        'temperature': tmp,
        'util.gpu': utilization_t.gpu,
        'util.memory': utilization_t.memory,
        'power.draw': milliWatts / 1000.0,
        'memory.used': memory_t.used,
        'timestamp': unix_time_millis
    }