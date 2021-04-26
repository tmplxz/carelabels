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
        # TODO configure via command line
        self.predict_samples = 50
        self.hide_amount = 0.2
        self.loaded = Result.from_file(self.f_name).data_dict
        self.X_train, X_test = train_test_split(self.loaded['sample_data'])
        self.X_test = X_test[:self.predict_samples]
        
        # hide some values
        self.X_hidden = np.zeros_like(self.X_test).astype(bool)
        idc1, idc2 = np.unravel_index(np.random.choice(np.arange(self.X_test.size), size=int(self.X_test.size * self.hide_amount), replace=False), self.X_test.shape)
        self.X_hidden[idc1, idc2] = True

    def get_train(self):
        return self.X_train

    def get_apply(self):
        data = self.X_test.copy()
        data[self.X_hidden] = -1
        return data

    def info(self):
        info = {}
        info['name'] = os.path.basename(self.f_name.split('.')[0])
        info['cl_name'] = ''.join(re.match(r'(.*)_nodes(\d*).*', os.path.basename(self.f_name)).groups()).capitalize()
        info['graph_type'] = get_graph_type(info['name'])
        info['states'] = int(np.max(self.loaded['states']))
        info['vertices'] = len(np.unique(self.loaded['edgelist']))
        info['T'] = self.loaded['T']
        info['edgelist'] = self.loaded['edgelist']
        info['weights'] = self.loaded['weights']
        info['mu'] = self.loaded['mu']
        info['seed'] = self.loaded['seed']
        info['gibbs_iterations'] = self.loaded['gibbs_iterations']
        info['max_delta'] = self.loaded['max_delta']
        info['predict_samples'] = self.predict_samples

        return info

    def evaluate(self, prediction):
        acc = np.count_nonzero(self.X_test[self.X_hidden] == prediction[self.X_hidden]) / np.count_nonzero(self.X_hidden)
        return {'Accuracy': acc}
