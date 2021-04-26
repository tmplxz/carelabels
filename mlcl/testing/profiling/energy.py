from .profiling import Profiling


class DummyEnergyProfiling(Profiling):

    def run(self, func):

        result = {
            'DURATION': 0.5,
            "package-0": 20.0,
            "dram": 1.0,
            "core": 21.0,
            "uncore": 0.0
        }

        return result, None


class PyRaplEnergyProfiling(Profiling):
    """
    This is based on https://github.com/wkatsak/py-rapl
    """

    def __init__(self):
        import rapl
        self.monitor = rapl.RAPLMonitor


    def run(self, func):

        inner_begin = self.monitor.sample()
        result = func()
        inner_end = self.monitor.sample()

        diff = inner_end - inner_begin
        prof = {'DURATION': diff.duration}

        # Extract package, dram and core values
        for d in diff.domains:
            domain = diff.domains[d]
            power = diff.average_power(package=domain.name)
            prof[domain.name] = power

            for sd in domain.subdomains:
                subdomain = domain.subdomains[sd]
                power = diff.average_power(package=domain.name, domain=subdomain.name)
                prof[subdomain.name] = power

        return prof, result
