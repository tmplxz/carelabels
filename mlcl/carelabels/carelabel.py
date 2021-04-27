import base64
import json
import os

import numpy as np
import xml.etree.ElementTree as ET

from mlcl.util import reformat_value


def flip_rating(info):
    if 'Accuracy' in info:
        info['Accuracy'][1] = 3 - info['Accuracy'][1]
    info['benchmark']['accuracy_rating'] = 3 - info['benchmark']['accuracy_rating']
    return info


cl_styles = [
    'st7',  # green hex
    'st19', # yellow hex
    'st12', # orange hex
    'st18', # red hex
    'st15', # green
    'st13', # yellow
    'st4',  # orange
    'st5',  # red
    'st16', # arrow grey
    'st23', # black
    'st8',  # white
    'st23', # n.a.
]


cl_nn_styles = [
    'st10', # hex green
    'st23', # hex yellow
    'st21', # hex orange
    'st22', # hex red
    'st19', # green
    'st16', # yellow
    'st18', # orange
    'st20', # red
    'st9',  # arrow false
    'st26', # black
    'st11',  # white
]


cl_text = [
    'A',
    'B',
    'C',
    'D',
    '?'
]


BADGE_MAP = {
    'Provides Uncertainty?': 'B_UNCERT',
    'Can be used on data streams?': 'B_STREAM',
    'Suitable for edge devices?': 'B_EDGE',
    'Is Robust?': 'B_ROBU'
}


T_UPWARDS = np.array([0,   -230])
# transform="matrix(1 0 0 1 0 -230)"
T_LEFT_UP = np.array([206, -115])
# transform="matrix(1 0 0 1 206 -115)" 


def align_badges(badges):
    # TODO improve the badge alignment
    transforms = {}
    for badge in BADGE_MAP:
        if badge == 'Provides Uncertainty?':
            transforms[badge] = "matrix(1 0 0 1 206 -115)"
        else:
            transforms[badge] = "matrix(1 0 0 1 0 0)"
    return transforms


def get_compliance_checkmark(theory, practice):
    if theory == practice:
        return cl_styles[4]
    return cl_styles[8]


class CareLabel:

    def __init__(self, label_info, designfile):
        self.label_info = flip_rating(label_info)
        self.designfile = os.path.join(os.path.dirname(__file__), designfile)

    def __str__(self):
        raise NotImplementedError

    def to_image(self, fname):
        raise NotImplementedError

    def generate_image(self, content_map, fname):
        tree = ET.parse(self.designfile)
        root = tree.getroot()
        for child in root.iter():
            if 'id' in child.attrib and child.attrib['id'] in content_map:
                if 'path' in child.tag or 'circle' in child.tag:
                    if isinstance(content_map[child.attrib['id']], str):
                        child.attrib['class'] = content_map[child.attrib['id']]
                    else: # tuple with style and transform information
                        style, transform = content_map[child.attrib['id']]
                        child.attrib['class'] = style
                        child.attrib['transform'] = transform
                    
                if 'text' in child.tag or 'tspan' in child.tag:
                    child.text = content_map[child.attrib['id']]

        tree.write(fname)


class MethodCareLabel(CareLabel):

    def __init__(self, label_info):
        super().__init__(label_info, 'carelabel_design.svg')
        self.styles = cl_styles
        self.label_info['badges'] = ['Provides Uncertainty?']

    def __str__(self):
        str_c = []
        str_c.append(f'CARE LABEL FOR {self.label_info["Name"].upper()}')
        str_c.append(self.label_info['Description'])
        str_c.append('Theoretical aspects:')
        for key, rating in self.label_info.items():
            if isinstance(rating, int) and not isinstance(rating, bool):
                str_c.append(f'    {key}: {cl_text[rating]}')
                if key in self.label_info['Fulfilled criteria']:
                    str_c.append('        Fulfilled criteria:')
                    for crit in self.label_info['Fulfilled criteria'][key]:
                        str_c.append(f'           {crit}')
                if key in self.label_info['Not fulfilled criteria']:
                    str_c.append('        Not fulfilled criteria:')
                    for crit in self.label_info['Not fulfilled criteria'][key]:
                        str_c.append(f'           {crit}')
                str_c.append('    ---------------------')
        str_c.append('Practical aspects:')
        # TODO print more content
        return '\n'.join(str_c)

    def to_image(self, fname):
        # construct map
        content_map = {
            # hex ratings, styles 0-3 and text 0-3
            'EXP_C': self.styles[self.label_info['Expressivity']],
            'EXP_R': cl_text[self.label_info['Expressivity']],
            'REL_C': self.styles[self.label_info['Reliability']],
            'REL_R': cl_text[self.label_info['Reliability']],
            'USA_C': self.styles[self.label_info['Usability']],
            'USA_R': cl_text[self.label_info['Usability']],
            'MEM_C': self.styles[self.label_info['Memory']],
            'MEM_R': cl_text[self.label_info['Memory']],
            'RUN_C': self.styles[self.label_info['Runtime']],
            'RUN_R': cl_text[self.label_info['Runtime']],
            # check arrows, either style 4 or 8
            'REL_A': self.styles[4] if all([check['success'] for check in self.label_info['Software tests'].values()]) and len(self.label_info['Software tests']) > 2 else self.styles[8],
            'RUN_A': self.styles[4] if self.label_info['Software tests']['runtime_complexity']['success'] else self.styles[8],
            'MEM_A': self.styles[4] if self.label_info['Software tests']['memory_complexity']['success'] else self.styles[8],
            # info texts
            'NAME': self.label_info["Name"],
            'DESCR': self.label_info["Description"],
            'PLAT': f'Platform: {self.label_info["Execution information"]["platform"]}',
            'SOFT': f'Software: {self.label_info["Execution information"]["software"]}',
            'DATA': f'Data set: {self.label_info["benchmark"]["name"]}',
            # performance measurements
            'RUNTIME': reformat_value(self.label_info["benchmark"]["runtime"], 's')
        }
        # benchmark measurements
        mem = reformat_value(self.label_info["benchmark"]["memory"], 'B', 1024)
        ene = reformat_value(self.label_info["benchmark"]["energy"], 'Ws')
        if self.label_info["benchmark"]["gpu_memory"] == 0: # only report cpu values
            benchmark = {
                'MEMORY': mem,
                'MEMORY1': '',
                'MEMORY2': '',
                'ENERGY': ene,
                'ENERGY1': '',
                'ENERGY2': '',
            }
        else: # report both values
            benchmark = {
                'MEMORY': '',
                'MEMORY1': f'CPU: {mem}',
                'MEMORY2': f'GPU: {reformat_value(self.label_info["benchmark"]["gpu_memory"], "B", 1024)}',
                'ENERGY': '',
                'ENERGY1': f'CPU: {ene}',
                'ENERGY2': f'GPU: {reformat_value(self.label_info["benchmark"]["gpu_energy"], "Ws")}',
            }
        content_map.update(benchmark)
        # performance labels, styles 4-7, consiting of several paths
        content_map.update({
            f'RUN_L{i}': self.styles[self.label_info["benchmark"]["runtime_rating"] + 4] for i in range(40)
        })
        content_map.update({
            f'MEM_L{i}': self.styles[self.label_info["benchmark"]["memory_rating"] + 4] for i in range(40)
        })
        content_map.update({
            f'ENE_L{i}': self.styles[self.label_info["benchmark"]["energy_rating"] + 4] for i in range(40)
        })
        # badges
        transforms = align_badges(self.label_info["badges"])
        for key, badge in BADGE_MAP.items():
            if key in self.label_info["badges"]:
                content_map.update({
                    f'{badge}_{i}': (self.styles[9], transforms[key]) for i in range(10)
                })
            else:
                content_map.update({
                    f'{badge}_{i}': (self.styles[10], transforms[key]) for i in range(10)
                })

        self.generate_image(content_map, fname)

class ModelCareLabel(CareLabel):

    def __init__(self, label_info):
        super().__init__(label_info, 'carelabel_nn_design.svg')
        self.styles = cl_nn_styles
    
    def __str__(self):
        return ''

    def to_image(self, fname):
        # construct map
        content_map = {
            # hex ratings, styles 0-3 and text 0-3, concrete values
            'ACC_C': self.styles[self.label_info['Accuracy'][1]],
            'ACC_R': cl_text[self.label_info['Accuracy'][1]],
            'ACC_T': f'({self.label_info["Accuracy"][0]} %)',
            'MEM_C': self.styles[self.label_info['Relative_memory'][1]],
            'MEM_R': cl_text[self.label_info['Relative_memory'][1]],
            'MEM_T': f'({self.label_info["Relative_memory"][0] / 1e6:4.2f} M Parameters)',
            'RUN_C': self.styles[self.label_info['Relative_runtime'][1]],
            'RUN_R': cl_text[self.label_info['Relative_runtime'][1]],
            'RUN_T': f'({self.label_info["Relative_runtime"][0] / 1e9:4.2f} Flops (G))',
            'ENE_C': self.styles[self.label_info['Train_energy'][1]],
            'ENE_R': cl_text[self.label_info['Train_energy'][1]],
            'ENE_T': f'({self.label_info["Train_energy"][0]:4.2f} kWh)',

            # info texts
            'NAME': self.label_info["Name"],
            'DESCR': self.label_info["Description"],
            'PLAT': f'Platform: {self.label_info["Execution information"]["platform"]}',
            'SOFT': f'Software: {self.label_info["Execution information"]["software"]}',
            'DATA': f'Data set: {self.label_info["benchmark"]["name"]}',
            'METR': f'Metrices: Top-1 Accuracy',
            # performance measurements
            'RUNTIME': reformat_value(self.label_info["benchmark"]["runtime"], 's'),
            'ACCURACY': reformat_value(self.label_info["benchmark"]["accuracy"], '%'),
            'DYN_PER': self.styles[self.label_info["Software tests"]['perturbationtest']['rating']],
            'DYN_PER_R': cl_text[self.label_info["Software tests"]['perturbationtest']['rating']],
            'DYN_COR': self.styles[self.label_info["Software tests"]['corruptiontest']['rating']],
            'DYN_COR_R': cl_text[self.label_info["Software tests"]['corruptiontest']['rating']],
            'DYN_NOI': self.styles[self.label_info["Software tests"]['noisetest']['rating']],
            'DYN_NOI_R': cl_text[self.label_info["Software tests"]['noisetest']['rating']]
        }
        # benchmark measurements
        mem = reformat_value(self.label_info["benchmark"]["memory"], 'B', 1024)
        ene = reformat_value(self.label_info["benchmark"]["energy"], 'Ws')
        if self.label_info["benchmark"]["gpu_memory"] == 0: # only report cpu values
            benchmark = {
                'MEMORY': mem,
                'MEMORY1': '',
                'MEMORY2': '',
                'ENERGY': ene,
                'ENERGY1': '',
                'ENERGY2': '',
            }
        else: # report both values
            benchmark = {
                'MEMORY': '',
                'MEMORY1': f'CPU: {mem}',
                'MEMORY2': f'GPU: {reformat_value(self.label_info["benchmark"]["gpu_memory"], "B", 1024)}',
                'ENERGY': '',
                'ENERGY1': f'CPU: {ene}',
                'ENERGY2': f'GPU: {reformat_value(self.label_info["benchmark"]["gpu_energy"], "Ws")}',
            }
        content_map.update(benchmark)
        # performance labels, styles 4-7, consisting of several paths
        accuracy_check = self.label_info["benchmark"]["accuracy"] + 1e-4 >= self.label_info["Accuracy"][0]
        content_map['ACC_L'] = self.styles[self.label_info["benchmark"]["accuracy_rating"] + 4]
        content_map['ACC_A'] = self.styles[4] if accuracy_check else self.styles[-1]
        content_map.update({
            f'RUN_L{i}': self.styles[self.label_info["benchmark"]["runtime_rating"] + 4] for i in range(40)
        })
        content_map.update({
            f'MEM_L{i}': self.styles[self.label_info["benchmark"]["memory_rating"] + 4] for i in range(40)
        })
        content_map.update({
            f'ENE_L{i}': self.styles[self.label_info["benchmark"]["energy_rating"] + 4] for i in range(40)
        })
        # badges
        transforms = align_badges(self.label_info["badges"])
        for key, badge in BADGE_MAP.items():
            if key in self.label_info["badges"]:
                content_map.update({
                    f'{badge}_{i}': (self.styles[9], transforms[key]) for i in range(10)
                })
            else:
                content_map.update({
                    f'{badge}_{i}': (self.styles[10], transforms[key]) for i in range(10)
                })

        self.generate_image(content_map, fname)