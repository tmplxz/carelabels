import json
import os
import time
import subprocess
import tempfile
import itertools
from collections import defaultdict

import numpy as np

from .reliability_checks import probability_recovery, runtime_complexity, memory_complexity


class Implementation_Profiling:

    def __init__(self, implementation, config, scaling, benchmark, logdir, repeats):
        self.implementation = implementation
        self.impl_config = config
        self.scaling = scaling
        self.benchmark = benchmark
        self.logdir = logdir
        self.repeats = repeats
        if not os.path.isdir(logdir):
            os.makedirs(logdir)

    def run(self):
        scaling_measurements = {}
        print(f'Running profiling on {len(self.scaling)} datafiles.')
        
        # write temp file with implementation config for the profiling subprocesses
        fd, tmp = tempfile.mkstemp('w')
        os.close(fd)
        with open(tmp, 'w') as tf:
            json.dump(self.impl_config, tf)
        if not os.path.isfile(tmp):
            raise RuntimeError(f'Could not find "{tmp}" as config file!')

        # run profiling for each dataset
        to_benchmark = self.scaling + [self.benchmark]
        for idx, df in enumerate(to_benchmark):

            # assess general information about dataset
            X_train, X_test, _, _, info = self.implementation.ds_class(df).data()
            bench_results = {
                'x_train_bytes': X_train.nbytes,
                'x_test_bytes': X_test.nbytes,
                'x_train_shape': X_train.shape,
                'x_test_shape': X_test.shape,
                'repeats': self.repeats,
                'start': time.time(),
            }
            bench_results.update(info)
            bench_results.update(self.implementation.get_info())
            impl_prefix = bench_results['config_id']
            del X_train
            del X_test

            profilings = ['memory', 'energy', 'runtime']
            # if self.impl_config['Platform'] == 'GPU':
                # profilings.append('gpu')

            # run profiling
            df_name = os.path.basename(df).split('.')[0]
            log_basename = f'{impl_prefix}_{df_name}'
            full_logname = os.path.join(self.logdir, f'{log_basename}.json')
            
            if os.path.isfile(full_logname): # if present, load current log state
                with open(full_logname, 'r', encoding='utf-8') as lf:
                    bench_results = json.load(lf)
                    if 'name' not in bench_results:
                        bench_results['name'] = info['name']
                    if bench_results['repeats'] < self.repeats:
                        bench_results['repeats'] = self.repeats

            for repeat in range(self.repeats):
                if 'TRAIN_RUNTIME_WCT' not in bench_results or len(bench_results['TRAIN_RUNTIME_WCT']) <= repeat: # otherwise this repeat was already run and logged
                    for profiling, train_apply in itertools.product(profilings, ['train', 'apply']):

                        logname = os.path.join(self.logdir, f'{log_basename}_{train_apply}_{profiling}_{repeat}.json')
                        t1 = time.time()
                        subprocess.run(["python", "-m" "mlcl.testing", "--profiling-method", profiling,
                                        "--implementation", self.implementation.__class__.__name__, "--config", tmp,
                                        "--datafile", df, "--train-apply", train_apply, "--log-file", logname])
                        t2 = time.time()
                        print(f'PROFILING {logname} TOOK {t2 - t1}')
                        # load results
                        with open(logname, 'r', encoding='utf-8') as lfile:
                            results = json.load(lfile)
                            entry_prefix = train_apply.upper() + '_' + profiling.upper()
                            for field, values in results.items():
                                key = entry_prefix + '_' + field
                                if key not in bench_results:
                                    bench_results[key] = []
                                bench_results[key].append(values)

            with open(full_logname, 'w', encoding='utf-8') as lf:
                # TODO make this better, maybe add a "log info" method to datasets
                if 'T' in bench_results:
                    del(bench_results['T'])
                if 'edgelist' in bench_results:
                    del(bench_results['edgelist'])
                if 'weights' in bench_results:
                    del(bench_results['weights'])
                if 'mu' in bench_results:
                    del(bench_results['mu'])
                if 'gibbs_iterations' in bench_results:
                    del(bench_results['gibbs_iterations'])
                if 'max_delta' in bench_results:
                    del(bench_results['max_delta'])
                json.dump(bench_results, lf)

            if idx == len(to_benchmark) - 1: # benchmark dataset
                benchmark_measurements = bench_results
            else: # scaling dataset for performance check
                scaling_measurements[info['name']] = bench_results

        # remove tmp config file
        os.remove(tmp)
        return scaling_measurements, benchmark_measurements


class Implementation_Reliability_Testing:

    def __init__(self, implementation, checks, implemented_checks=None):
        self.implementation = implementation
        self.checks = checks
        if implemented_checks is None: # defaults
            implemented_checks = [probability_recovery, runtime_complexity, memory_complexity]
        self.execute = {check.__name__: check for check in implemented_checks if check.__name__ in self.checks.values()}
        not_implemented = [check for check in self.checks.values() if check not in self.execute.keys()]
        if len(not_implemented) > 0:
            raise RuntimeError(f'Could not find implementations for checks: {" ".join(not_implemented)}.')

    def run(self, args=None):
        def_args = defaultdict(lambda: None)
        if args is not None:
            def_args.update(args)
        reliability_checks = {}
        for name, check in self.execute.items():
            reliability_checks[name] = check(self.implementation, def_args[name])
        return reliability_checks
