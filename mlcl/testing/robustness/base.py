import os
import json


class RobustnessTest:

    def __init__(self, implementation, benchmark, logdir):
        self.implementation = implementation
        self.benchmark = benchmark
        impl_id = self.implementation.get_info()['config_id']
        self.logname = os.path.join(logdir, f'{impl_id}_robustness_{self.id()}.json')

    def id(self):
        return self.__class__.__name__.lower()

    def run(self):
        if os.path.isfile(self.logname):
            with open(self.logname, 'r') as lfile:
                result = json.load(lfile)
        else:
            result = self.run_test()
            with open(self.logname, 'w') as lfile:
                json.dump(result, lfile)
        return result

    def run_test(self):
        return {
            'score': 0.0,
            'description': 'Default',
            'details': 'Default'
        }
