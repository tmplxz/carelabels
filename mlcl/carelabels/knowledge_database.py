import json
import os
import re

import numpy as np

from .carelabel import MethodCareLabel, ModelCareLabel
from mlcl.util import extract_benchmark_results, check_scale


default_scales = {
    'runtime_time':       [     1e3,        1,          1e-3    ],
    'relative_runtime':   [     4e9,        2e9,        1e9     ],
    'memory_bytes':       [     750e6,      500e6,      250e6   ],
    'relative_memory':    [     100e6,      50e6,       10e6    ],
    'energy_Ws':          [     1e3,        1,          1e-3    ],
    'train_energy':       [     20,         10,         5       ],
    'criteria_fulfilled': [     0.2,        0.5,        0.75    ],
    'accuracy':           [     70,         69.5,       69      ],
    'corruptiontest':     [     0.95,       0.9,        0.8     ],
    'perturbationtest':   [     0.085,      0.07,       0.065   ],
    'noisetest':          [     2e-7,       3e-7,       5e-7    ],
}


def check_fulfilled(all_criteria, bins):
    rating = {}
    fulfilled_criteria = {}
    not_fulfilled_crit = {}
    for category, criteria in all_criteria.items():
        fulfilled_criteria[category] = []
        not_fulfilled_crit[category] = []
        for criterium, fulfilled in criteria.items():
            if fulfilled:
                fulfilled_criteria[category].append(criterium)
            else:
                not_fulfilled_crit[category].append(criterium)
        perc_fulfilled = len(fulfilled_criteria[category]) / len(criteria)
        rating[category] = check_scale(perc_fulfilled, bins)
        # print(f'{category:<15} - {perc_fulfilled:5.2f} - {rating[category]} - {bins}')
        if category == 'Meta':
            rating[category] = -1
    return rating, fulfilled_criteria, not_fulfilled_crit


def update_criteria(crit1, crit2):
    for cat, crit in crit1.items():
        crit.update(crit2[cat])


def recursive_criteria_parse(criteria_node, criteria_value, lookup):
    criteria = {'Expressivity': {}, 'Reliability': {}, 'Usability': {}, 'Resources': {}, 'Meta': {}}
    if isinstance(criteria_value, str):
        criteria[criteria_value][criteria_node] = criteria_node in lookup
        return criteria
    else:
        if criteria_node is None: # top level
            for node, value in criteria_value.items():
                subcrit = recursive_criteria_parse(node, value, lookup)
                update_criteria(criteria, subcrit)
                
        else: # meta criterium, so check if applicable and go into depth
            for c in lookup:
                if isinstance(c, dict) and criteria_node in c.keys():
                    criteria['Meta'][criteria_node] = True
                    for node, value in criteria_value.items():
                        subcrit = recursive_criteria_parse(node, value, list(c.values())[0])
                        update_criteria(criteria, subcrit)
                    break
            else:
                criteria['Meta'][criteria_node] = False
        return criteria


class ExpertKnowledgeDatabase:

    def __init__(self, custom_scales=None, implemented_checks=None):
        self.scales = default_scales
        if custom_scales is not None:
            self.scales.update(custom_scales)
        self.checks = implemented_checks
        self.all = {}
        criteria_dir = os.path.join(os.path.dirname(__file__), 'criteria')
        for fname in os.listdir(criteria_dir):
            try:
                name = re.match('criteria_(.*).json', fname).group(1)
                with open(os.path.join(criteria_dir, fname), 'r', encoding='utf-8') as clf:
                    self.all[name] = json.load(clf)
            except AttributeError:
                pass # this file does not store any criteria
        self.label_info = {}
        if self.checks is None: # load default criteria check map
            with open(os.path.join(criteria_dir, 'checks.json'), 'r', encoding='utf-8') as ch:
                self.checks = json.load(ch)

    def parse_config(self, cl_file, config):
        with open(cl_file, 'r', encoding='utf-8') as clf:
            cl_content = json.load(clf)
        if 'dnn' in cl_file:
            self.label_class = ModelCareLabel
            self.checks = {}
            return self.parse_model_config(cl_content, config)
        else:
            self.label_class = MethodCareLabel
            return self.parse_method_config(cl_content, config)

    def parse_model_config(self, cl_content, config):
        if config['model'] not in cl_content:
            raise RuntimeError(f'Invalid config name "{config["model"]}", please pass one of the following model names: {", ".join(list(cl_content.keys()))}')
        for field, value in cl_content[config['model']].items():
            if field in self.scales:
                self.label_info[field.capitalize()] = [value, check_scale(value, self.scales[field])]
            else:
                self.label_info[field.capitalize()] = value
        if cl_content[config['model']]['relative_memory'] < 10e6:
            self.label_info['badges'] = ['Suitable for edge devices?']
        else:
            self.label_info['badges'] = []
        return config

    def parse_method_config(self, cl_content, config):
        # load content from care label file

        self.label_info['Name'] = cl_content['name']
        self.label_info['Description'] = cl_content['description']
        self.label_info['Runtime'] = cl_content['runtime']
        self.label_info['Memory'] = cl_content['memory']

        # basic criteria
        criteria = recursive_criteria_parse(None, self.all['model'], cl_content['criteria'])

        for c_name, c_values in cl_content['configurable'].items():
            # find configuration
            valid = list(c_values.keys())
            if c_name in config:
                if config[c_name] not in c_values.keys():
                    raise RuntimeError(f'Invalid config "{config[c_name]}" for "{c_name}". Valid options: {" ".join(valid)}.')
                c_option_name = config[c_name]
            else: # first is default
                c_option_name = valid[0]
            # update config, and check for config specific updates of care label
            config[c_name] = c_option_name
            if 'criteria' in c_values[c_option_name]:
                update_criteria(criteria, recursive_criteria_parse(None, self.all[c_name.lower()], c_values[c_option_name]['criteria']))
            if 'runtime' in c_values[c_option_name]:
                self.label_info['Runtime'] = max(self.label_info['Runtime'], c_values[c_option_name]['runtime'])
            if 'memory' in c_values[c_option_name]:
                self.label_info['Memory'] = max(self.label_info['Memory'], c_values[c_option_name]['memory'])
            if 'name' in c_values[c_option_name]:
                self.label_info['Name'] = self.label_info['Name'].replace(f'${c_name}', c_values[c_option_name]['name'])
            if 'description' in c_values[c_option_name]:
                self.label_info['Description'] = self.label_info['Description'].replace(f'${c_name}', c_values[c_option_name]['description'])
        
        rating, fulfilled, not_fulfilled = check_fulfilled(criteria, self.scales['criteria_fulfilled'])
        self.label_info.update(rating)
        self.label_info['Fulfilled criteria'] = fulfilled
        self.label_info['Not fulfilled criteria'] = not_fulfilled
        # TODO refine this, derive from criteria
        self.label_info['badges'] = ['Provides Uncertainty?']
        # find which reliability checks need to be run
        for crit, _ in list(self.checks.items()):
            if not any([crit in ful for ful in fulfilled.values()]):
                del self.checks[crit]
        return config

    def rate_software_tests(self, test_results):
        for name, result in test_results.items():
            if name in self.scales:
                result['rating'] = check_scale(result['score'], self.scales[name])
        return test_results

    def generate_label(self, benchmark_measurements=None, software_tests=None, execution_info=None):
        if self.label_info is None:
            raise RuntimeError('Please run "parse_config" before generating a care label!')

        self.label_info['benchmark'] = extract_benchmark_results(benchmark_measurements, self.scales)
        
        self.label_info['Software tests'] = self.rate_software_tests(software_tests)
        self.label_info['Execution information'] = execution_info

        # generate final care label
        cl = self.label_class(self.label_info)

        return cl
