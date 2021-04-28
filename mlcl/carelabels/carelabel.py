import base64
import json
import os

import numpy as np
import xml.etree.ElementTree as ET

from mlcl.util import reformat_value


CL_MET_STYLES = [
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


CL_MOD_STYLES = [
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


CL_RATINGS = [
    'A',
    'B',
    'C',
    'D',
    '?'
]


# badge information
BADGE_MAP = {
    'Provides Uncertainty?': 'B_UNCERT',
    'Can be used on data streams?': 'B_STREAM',
    'Suitable for edge devices?': 'B_EDGE',
    'Is Robust?': 'B_ROBU'
}
# transform="matrix(1 0 0 1 x y)"
T_NEUTR  = np.array([0, 0])
T_DOWN   = np.array([0, 234])
T_UP     = np.array([0, -234])
T_R_DOWN = np.array([206, 117])
T_L_DOWN = np.array([-206, 117])


def flip_rating(info):
    if 'Accuracy' in info:
        info['Accuracy'][1] = 3 - info['Accuracy'][1]
    info['benchmark']['accuracy_rating'] = 3 - info['benchmark']['accuracy_rating']
    return info


def process_badges(badges, colorstyles, pos_normalization, layouts):
    badge_contents = {}
    c = 0
    fulf = len(badges)
    for badge, badge_id in BADGE_MAP.items():
        if badge in badges:
            color = colorstyles[9]
            t_x, t_y = layouts[fulf][c] + pos_normalization[badge]
            c += 1
        else: # default position for unused badges
            color = colorstyles[10]
            t_x, t_y = layouts[fulf][-1] + pos_normalization[badge]
        badge_contents.update({
            f'{badge_id}_{i}': (color, f"matrix(1 0 0 1 {t_x} {t_y})") for i in range(10)
        })
    return badge_contents


def generate_image(designfile, content_map, fname):
    tree = ET.parse(designfile)
    root = tree.getroot()
    for child in root.iter():
        if 'id' in child.attrib and child.attrib['id'] in content_map:
            if 'path' in child.tag or 'circle' in child.tag:
                if isinstance(content_map[child.attrib['id']], str):
                    child.attrib['class'] = content_map[child.attrib['id']]
                else: # list with style and transform information
                    style, transform = content_map[child.attrib['id']]
                    child.attrib['class'] = style
                    child.attrib['transform'] = transform
                
            if 'text' in child.tag or 'tspan' in child.tag:
                child.text = content_map[child.attrib['id']]

    tree.write(fname)


class CareLabel:

    def __init__(self, label_info, designfile):
        self.label_info = flip_rating(label_info)
        self.designfile = os.path.join(os.path.dirname(__file__), designfile)

    def __str__(self):
        raise NotImplementedError

    def to_image(self, fname):
        raise NotImplementedError


class MethodCareLabel(CareLabel):

    def __init__(self, label_info):
        super().__init__(label_info, 'carelabel_design.svg')
        self.styles = CL_MET_STYLES
        # shifts all badges in the care label to the same position (top hexagon)
        self.badge_normalization = {
            'Suitable for edge devices?': T_NEUTR,
            'Provides Uncertainty?': T_R_DOWN + 2 * T_UP,
            'Can be used on data streams?': T_L_DOWN + 2 * T_UP,
            'Is Robust?': T_NEUTR
        }
        # the n positions for n activated badges, + default position for unused badges
        self.badge_layouts = {
            3: [T_NEUTR, T_DOWN + T_L_DOWN, T_DOWN + T_R_DOWN, T_L_DOWN],
            2: [T_DOWN + T_L_DOWN, T_R_DOWN, T_NEUTR],
            1: [T_DOWN, T_NEUTR],
            0: [T_NEUTR]
        }

    def __str__(self):
        str_c = []
        str_c.append(f'CARE LABEL FOR {self.label_info["Name"].upper()}')
        str_c.append(self.label_info['Description'])
        str_c.append('Theoretical aspects:')
        for key, rating in self.label_info.items():
            if isinstance(rating, int) and not isinstance(rating, bool):
                str_c.append(f'    {key}: {CL_RATINGS[rating]}')
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
            'EXP_R': CL_RATINGS[self.label_info['Expressivity']],
            'REL_C': self.styles[self.label_info['Reliability']],
            'REL_R': CL_RATINGS[self.label_info['Reliability']],
            'USA_C': self.styles[self.label_info['Usability']],
            'USA_R': CL_RATINGS[self.label_info['Usability']],
            'MEM_C': self.styles[self.label_info['Memory']],
            'MEM_R': CL_RATINGS[self.label_info['Memory']],
            'RUN_C': self.styles[self.label_info['Runtime']],
            'RUN_R': CL_RATINGS[self.label_info['Runtime']],
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
        content_map.update(process_badges(self.label_info["badges"], self.styles,
                                          self.badge_normalization, self.badge_layouts))

        generate_image(self.designfile, content_map, fname)

class ModelCareLabel(CareLabel):

    def __init__(self, label_info):
        super().__init__(label_info, 'carelabel_nn_design.svg')
        self.styles = CL_MOD_STYLES
        # shifts all badges in the care label to the same position (top hexagon)
        self.badge_normalization = {
            'Suitable for edge devices?': T_NEUTR,
            'Is Robust?': T_R_DOWN + T_UP,
            'Provides Uncertainty?': T_NEUTR,
            'Can be used on data streams?': T_NEUTR
        }
        # the n positions for n activated badges, + default position for unused badges
        self.badge_layouts = {
            3: [T_NEUTR, T_L_DOWN, T_R_DOWN, T_NEUTR],
            2: [T_L_DOWN, T_R_DOWN, T_NEUTR],
            1: [T_NEUTR, T_L_DOWN],
            0: [T_NEUTR]
        }
    
    def __str__(self):
        return ''

    def to_image(self, fname):
        # construct map
        content_map = {
            # hex ratings, styles 0-3 and text 0-3, concrete values
            'ACC_C': self.styles[self.label_info['Accuracy'][1]],
            'ACC_R': CL_RATINGS[self.label_info['Accuracy'][1]],
            'ACC_T': f'({self.label_info["Accuracy"][0]} %)',
            'MEM_C': self.styles[self.label_info['Relative_memory'][1]],
            'MEM_R': CL_RATINGS[self.label_info['Relative_memory'][1]],
            'MEM_T': f'({self.label_info["Relative_memory"][0] / 1e6:4.2f} M Parameters)',
            'RUN_C': self.styles[self.label_info['Relative_runtime'][1]],
            'RUN_R': CL_RATINGS[self.label_info['Relative_runtime'][1]],
            'RUN_T': f'({self.label_info["Relative_runtime"][0] / 1e9:4.2f} Flops (G))',
            'ENE_C': self.styles[self.label_info['Train_energy'][1]],
            'ENE_R': CL_RATINGS[self.label_info['Train_energy'][1]],
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
            'DYN_PER_R': CL_RATINGS[self.label_info["Software tests"]['perturbationtest']['rating']],
            'DYN_COR': self.styles[self.label_info["Software tests"]['corruptiontest']['rating']],
            'DYN_COR_R': CL_RATINGS[self.label_info["Software tests"]['corruptiontest']['rating']],
            'DYN_NOI': self.styles[self.label_info["Software tests"]['noisetest']['rating']],
            'DYN_NOI_R': CL_RATINGS[self.label_info["Software tests"]['noisetest']['rating']]
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
        content_map['ACC_A'] = self.styles[4] if accuracy_check else self.styles[8]
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
        content_map.update(process_badges(self.label_info["badges"], self.styles,
                                          self.badge_normalization, self.badge_layouts))

        generate_image(self.designfile, content_map, fname)
