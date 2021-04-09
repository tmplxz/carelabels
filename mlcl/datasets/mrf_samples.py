import os
import pickle
import re

from sklearn.model_selection import train_test_split
import numpy as np

from .base import DatasetBase


def get_graph_type(df):
    if 'chain' in os.path.basename(df):
        return 'chain'
    elif 'grid' in os.path.basename(df):
        return 'grid'
    else:
        raise RuntimeError('Unknown graph (for the moment...)')


class Result(object):

    def __init__(self, weights, edgelist,
                 states, sample_data,
                 seed, mu, gibbs_iterations,
                 suff_stats,
                 max_delta=0.0, T=1):
        self.data_dict = {
            'weights': weights,
            'edgelist': edgelist,
            'states': states,
            'sample_data': sample_data,
            'seed': seed,
            'gibbs_iterations': gibbs_iterations,
            'mu': mu,
            'suff_stats': suff_stats,
            'max_delta': max_delta,
            'T': T
        }

    def to_file(self, f_dir, f_file):
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)

        f_name = os.path.join(f_dir, f_file)
        with open(f_name, 'wb') as fd:
            pickle.dump(self.data_dict, fd)

    @classmethod
    def from_file(cls, path):
        with open(path, 'rb') as fd:
            data = pickle.load(fd)
        return cls(**data)


class MRF_Samples(DatasetBase):

    def __init__(self, cfg):
        self.f_name = cfg

    def data(self):
        loaded = Result.from_file(self.f_name).data_dict
        info = {}
        info['name'] = os.path.basename(self.f_name.split('.')[0])
        info['cl_name'] = ''.join(re.match('(.*)_nodes(\d*).*', os.path.basename(self.f_name)).groups()).capitalize()
        info['graph_type'] = get_graph_type(info['name'])
        info['states'] = int(np.max(loaded['states']))
        info['vertices'] = len(np.unique(loaded['edgelist']))
        info['T'] = loaded['T']
        info['edgelist'] = loaded['edgelist']
        info['weights'] = loaded['weights']
        info['mu'] = loaded['mu']
        info['seed'] = loaded['seed']
        info['gibbs_iterations'] = loaded['gibbs_iterations']
        info['max_delta'] = loaded['max_delta']
        X_train, X_test = train_test_split(loaded['sample_data'])
        y_train, y_test = None, None

        return X_train, X_test, y_train, y_test, info
