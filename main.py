import argparse
import json
import os

from mlcl.testing import ImplementationProfiling, ImplementationReliabilityTesting, ImplementationRobustnessTesting
from mlcl.util import prepare_dataset_list, init_implementation
from mlcl.carelabels import ExpertKnowledgeDatabase


def main(implementation, config_args, scaling_data, benchmark, log_dir, repeats):

    # init implementation
    implementation = init_implementation(implementation)
    
    # init knowledge database
    cl_database = ExpertKnowledgeDatabase()
    
    # parse config for the given care label
    config = cl_database.parse_config(implementation.cl_file, config_args)
    
    # prepare implementation with config
    implementation.prepare(config)

    scaling, benchmark = prepare_dataset_list(scaling_data, benchmark)

    # run performance bound checks / profiling
    tester = ImplementationProfiling(implementation, config, scaling, benchmark, log_dir, repeats)
    scaling_measurements, benchmark_measurements = tester.run()

    # run robustness checks OR reliability checks
    if len(cl_database.checks) < 1:
        tester = ImplementationRobustnessTesting(implementation, benchmark, log_dir)
    else:
        reliability_args = {
            'runtime_complexity': {
                'measurements': scaling_measurements,
                'theoretical': cl_database.label_info['Runtime']
            },
            'memory_complexity': {
                'measurements': scaling_measurements,
                'theoretical': cl_database.label_info['Memory']
            }
        }
        tester = ImplementationReliabilityTesting(implementation, cl_database.checks, reliability_args)
    software_tests = tester.run()
    
    return cl_database.generate_label(benchmark_measurements, software_tests, implementation.get_meta_info())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--implementation', default='PretrainedImageNetDNN', help='Implementation that shall be used.')
    parser.add_argument('--config', default='cfg/mobilenet_gpu.json')
    parser.add_argument('--repeats', default=1, type=int, help='Number of profiling repeats.')
    parser.add_argument('--log-dir', default='logs', help='Directory to store results.')
    parser.add_argument('--benchmark', default='/data/imagenet', help='Benchmark dataset to use.')
    parser.add_argument('--scaling-data', default='', help='Directory with data files.')
    parser.add_argument('--out', default='out.svg', help='Name of care label SVG that shall be generated.')

    args = parser.parse_args()

    if args.config.endswith('.json'): # config stored in json file
        with open(args.config, 'r', encoding='utf-8') as clf:
            config_args = json.load(clf)
    else:
        config_args = args.config

    label = main(args.implementation, config_args, args.scaling_data, args.benchmark, args.log_dir, args.repeats)
    print(label)
    label.to_image(args.out)
