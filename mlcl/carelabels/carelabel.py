import base64
import json
import os

import numpy as np
import xml.etree.ElementTree as ET

from mlcl.util import reformat_value


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
    'st23', # n.a.
]


cl_text = [
    'A',
    'B',
    'C',
    'D',
    '?'
]


def get_compliance_checkmark(theory, practice):
    if theory == practice:
        return cl_styles[4]
    return cl_styles[8]


class Care_Label:

    def __init__(self, label_info, label_details):
        self.label_info = label_info
        self.label_details = label_details

    def as_json(self):
        raise NotImplementedError

    def __str__(self):
        str_c = []
        str_c.append(f'CARE LABEL FOR {self.label_info["Name"].upper()}')
        str_c.append(self.label_info['Description'])
        str_c.append('Theoretical aspects:')
        for key, rating in self.label_info.items():
            if isinstance(rating, int) and not isinstance(rating, bool):
                str_c.append(f'    {key}: {cl_text[rating]}')
                if key in self.label_details['Fulfilled criteria']:
                    str_c.append('        Fulfilled criteria:')
                    for crit in self.label_details['Fulfilled criteria'][key]:
                        str_c.append(f'           {crit}')
                if key in self.label_details['Not fulfilled criteria']:
                    str_c.append('        Not fulfilled criteria:')
                    for crit in self.label_details['Not fulfilled criteria'][key]:
                        str_c.append(f'           {crit}')
                str_c.append('    ---------------------')
        str_c.append('Practical aspects:')
        # TODO print more content
        return '\n'.join(str_c)

    def to_image(self, fname=None):
        # construct map
        content_map = {
            # hex ratings, styles 0-3 and text 0-3
            'EXP_C': cl_styles[self.label_info['Expressivity']],
            'EXP_R': cl_text[self.label_info['Expressivity']],
            'REL_C': cl_styles[self.label_info['Reliability']],
            'REL_R': cl_text[self.label_info['Reliability']],
            'USA_C': cl_styles[self.label_info['Usability']],
            'USA_R': cl_text[self.label_info['Usability']],
            'MEM_C': cl_styles[self.label_info['Memory']],
            'MEM_R': cl_text[self.label_info['Memory']],
            'RUN_C': cl_styles[self.label_info['Runtime']],
            'RUN_R': cl_text[self.label_info['Runtime']],
            # check arrows, either style 4 or 8
            'REL_A': cl_styles[4] if all([check['success'] for check in self.label_details['Reliability checks'].values()]) and len(self.label_details['Reliability checks']) > 2 else cl_styles[8],
            'RUN_A': cl_styles[4] if self.label_details['Reliability checks']['runtime_complexity']['success'] else cl_styles[8],
            'MEM_A': cl_styles[4] if self.label_details['Reliability checks']['memory_complexity']['success'] else cl_styles[8],
            # info texts
            'NAME': self.label_info["Name"],
            'DESCR': self.label_info["Description"],
            'PLAT': f'Platform: {self.label_info["execution_info"]["platform"]}',
            'SOFT': f'Software: {self.label_info["execution_info"]["software"]}',
            'DATA': f'Dataset: {self.label_info["benchmark"]["name"]}',
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
            f'RUN_L{i}': cl_styles[self.label_info["benchmark"]["runtime_rating"] + 4] for i in range(40)
        })
        content_map.update({
            f'MEM_L{i}': cl_styles[self.label_info["benchmark"]["memory_rating"] + 4] for i in range(40)
        })
        content_map.update({
            f'ENE_L{i}': cl_styles[self.label_info["benchmark"]["energy_rating"] + 4] for i in range(40)
        })
        designfile = os.path.join(os.path.dirname(__file__), 'carelabel_design.svg')
        tree = ET.parse(designfile)
        root = tree.getroot()
        for idx, child in enumerate(root.iter()):
            if 'id' in child.attrib and child.attrib['id'] in content_map:
                if 'path' in child.tag or 'circle' in child.tag:
                    child.attrib['class'] = content_map[child.attrib['id']]
                if 'text' in child.tag or 'tspan' in child.tag:
                    child.text = content_map[child.attrib['id']]

        if fname is not None:
            tree.write(fname)
