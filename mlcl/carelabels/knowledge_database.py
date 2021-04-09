import json
import os
import re

import numpy as np

from .carelabel import Care_Label
from mlcl.util import extract_benchmark_results, check_scale


default_runtime_scale = [ # seconds
    1e3,
    1,
    1e-3
]


default_memory_scale = [ # bytes
    750e6,
    500e6,
    250e6
]


default_energy_scale = [ # m 
    1e3,
    1,
    1e-3
]


def check_fulfilled(all_criteria, bins=[0.2, 0.5, 0.75]):
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
        if category == 'Meta':
            rating[category] = -1
    return rating, fulfilled_criteria, not_fulfilled_crit


def update_criteria(crit1, crit2):
    for cat, crit in crit1.items():
        crit.update(crit2[cat])


def recursive_criteria_parse(criteria_node, criteria_value, lookup):
    criteria = {'Expressivity': {}, 'Reliability': {}, 'Usability': {}, 'Meta': {}}
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


class Expert_Knowledge_Database:

    def __init__(self, runtime_scale=None, memory_scale=None, energy_scale=None, implemented_checks=None):
        if runtime_scale is None:
            runtime_scale = default_runtime_scale
        if memory_scale is None:
            memory_scale = default_memory_scale
        if energy_scale is None:
            energy_scale = default_energy_scale
        self.runtime_scale = runtime_scale
        self.memory_scale = memory_scale
        self.energy_scale = energy_scale
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
        self.label_details = {}
        if self.checks is None: # load default criteria check map
            with open(os.path.join(criteria_dir, 'checks.json'), 'r', encoding='utf-8') as ch:
                self.checks = json.load(ch)

    def parse_config(self, cl_file, config):
        # load content from care label file
        with open(cl_file, 'r', encoding='utf-8') as clf:
            cl_content = json.load(clf)

        self.label_info['Name'] = cl_content['name']
        self.label_info['Description'] = cl_content['description']
        self.label_info['Runtime'] = cl_content['runtime']
        self.label_info['Memory'] = cl_content['memory']

        # basic criteria
        criteria = recursive_criteria_parse(None, self.all['model'], cl_content['criteria'])

        required = {}
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
        
        rating, fulfilled, not_fulfilled = check_fulfilled(criteria)
        self.label_info.update(rating)
        self.label_details['Fulfilled criteria'] = fulfilled
        self.label_details['Not fulfilled criteria'] = not_fulfilled
        # find which reliability checks need to be run
        for crit, check in list(self.checks.items()):
            if not any([crit in ful for ful in fulfilled.values()]):
                del self.checks[crit]
        return config

    def generate_label(self, benchmark_measurements=None, reliability_checks=None, execution_info=None):
        if self.label_info is None:
            raise RuntimeError('Please run "parse_config" before generating a care label!')
        if execution_info is None:
            execution_info = {
                'Name': 'Default',
                'Language': 'Python',
                'Platform': 'Default',
            }

        self.label_info['benchmark'] = extract_benchmark_results(benchmark_measurements, 
                                                                 self.runtime_scale,
                                                                 self.memory_scale,
                                                                 self.energy_scale)
        
        self.label_details['Reliability checks'] = reliability_checks
        self.label_info['execution_info'] = execution_info

        # generate final care label
        cl = Care_Label(self.label_info, self.label_details)

        return cl
