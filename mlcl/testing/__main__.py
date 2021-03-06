import json
import argparse
import sys
import traceback

from mlcl.testing.profiling.memory import MemoryProfiling
from mlcl.testing.profiling.runtime import RuntimeProfiling
from mlcl.util import init_implementation


if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--profiling-method', help='Type of profiling that shall be run.')
        parser.add_argument('--implementation', help='Implemented method.')
        parser.add_argument('--config', help='Config for implementation.')
        parser.add_argument('--datafile', help='Data for profiling.')
        parser.add_argument('--train-apply', help='Indicates whether to run training or application.')
        parser.add_argument('--log-file', help='Name of log file to write.')

        args = parser.parse_args()

        # load profiling
        if args.profiling_method == 'energy':
            try:
                from mlcl.testing.profiling.energy import PyRaplEnergyProfiling
                profiling = PyRaplEnergyProfiling()
            except ModuleNotFoundError:
                # TODO add more Energy profilers here, e.g. for Windows
                from mlcl.testing.profiling.energy import DummyEnergyProfiling
                profiling = DummyEnergyProfiling()
        elif args.profiling_method == 'runtime':
            profiling = RuntimeProfiling()
        elif args.profiling_method == 'memory':
            profiling = MemoryProfiling()
        elif args.profiling_method == 'gpu':
            from mlcl.testing.profiling.gpu_prof import GpuMonitoringProcess
            profiling = GpuMonitoringProcess()
        else:
            raise RuntimeError(f'Profiling {args.profiling_method} not implemented!')
        
        # load implementation and data
        implementation = init_implementation(args.implementation)

        with open(args.config, 'r', encoding='utf-8') as clf:
            config = json.load(clf)
        
        implementation.prepare(config)
        data_handler = implementation.ds_class(args.datafile)
        data_info = data_handler.info()

        data_info['logfile'] = args.log_file

        if args.train_apply == 'train':
            data_train = data_handler.get_train()
            func = lambda: implementation.train(data_train, data_info)
        elif args.train_apply == 'apply':
            data_apply = data_handler.get_apply()
            func = lambda: implementation.apply(data_apply, data_info)
        else:
            raise RuntimeError(f'"{args.train_apply}" is not valid, please pass "train" or "apply"!')

        # run and log profiling
        prof, result = profiling.run(func)
        if result is not None:
            prof.update(data_handler.evaluate(result))
        success = 0

    except Exception as exc:
        prof = {
            'ERROR': str(exc),
            'DETAILS': str(traceback.format_exc())
        }
        success = 1

    with open(args.log_file, 'w', encoding='utf-8') as lf:
        json.dump(prof, lf)

    sys.exit(success)