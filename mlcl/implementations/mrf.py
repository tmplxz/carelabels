import os

import numpy as np
import pxpy as px

from mlcl.implementations.base import BaseImplementation
from mlcl.datasets import MRF_Samples


def split_marginals(model, mu):
    mu_shape = mu.shape[0]
    dim = model.dim
    cut_off = mu_shape - dim
    return mu[:cut_off], mu[cut_off:]


class PXMRF(BaseImplementation):

    def __init__(self, train_iters=50, predict_iters=50, infer_iters=10, use_gpu=False):
        super().__init__()
        self.train_iters = train_iters
        self.predict_iters = predict_iters
        self.gpu = use_gpu
        self.gpu_id = 'cpu' # might be overridden to 'gpu' if configured accordingly
        self.infer_iters = infer_iters
        self.cl_file = 'mlcl/carelabels/cl_mrf.json'
        self.ds_class = MRF_Samples
        self.inference = None
        self.mode = None
        self.maxiter = 1000000
        self.stop_counter = 0
        self.grace_period = 100
        self.prev_obj = np.infty

    def get_model_path(self, logname, replace_train_apply=False):
        mname = logname.split('.')[0] + '.px'
        if replace_train_apply:
            dirn = os.path.dirname(mname)
            base = os.path.basename(mname)
            base = base.replace('apply', 'train')
            mname = os.path.join(dirn, base)
        return mname

    def train(self, data, ds_info=None):
        # should only have discrete positive entries
        assert (data.min() >= 0)
        assert (issubclass(data.dtype.type, np.integer))
        modelpath = self.get_model_path(ds_info['logfile'])

        if 'itype' in self.additional_PX_args:
            graph = px.create_graph(ds_info['edgelist'], itype=self.additional_PX_args['itype'])
        else:
            graph = px.create_graph(ds_info['edgelist'])
        if self.mode == px.ModelType.strf_linear:
            self.additional_PX_args['T'] = ds_info['T']
        else:
            if 'T' in self.additional_PX_args:
                del self.additional_PX_args['T']

        model = px.train(data, iters=self.train_iters, mode=self.mode, graph=graph, inference=self.inference, infer_iters=self.infer_iters, infer_epsilon=-1, **self.additional_PX_args)

        # Extract learned weights
        self.weights = model.weights
        model.save(modelpath)
        model.graph.delete()
        model.delete()

    def apply(self, data, ds_info=None):
        # Check if fit had been called
        assert (data.min() >= 0)
        assert (issubclass(data.dtype.type, np.integer))
        modelpath = self.get_model_path(ds_info['logfile'], replace_train_apply=True)
        model = px.load_model(modelpath)

        eps_pre = px.read_register('EPS')
        px.write_register('EPS', px.integer_from_float(-1.0))

        model.predict(data, inference=self.inference, iterations=self.predict_iters)

        px.write_register('EPS', eps_pre)

        # Delete model
        model.graph.delete()
        model.delete()

        return data

    def prepare(self, args):
        self.config = args
        self.additional_PX_args = {}
        # evaluate additional config
        if 'Additional' in self.config:
            for key, val in self.config['Additional'].items():
                if hasattr(self, key):
                    setattr(self, key, val)
        # modeltype
        if self.config['Type'] == 'MRF':
            self.mode = px.ModelType.mrf
        # inference
        if self.config['Inference'] == 'Belief_Propagation':
            self.inference = px.InferenceType.belief_propagation
            self.inference_str = 'BP'
        elif self.config['Inference'] == 'Junction_Tree':
            self.inference = px.InferenceType.junction_tree
            self.inference_str = 'JT'
        if self.gpu:
            self.inference = px.InferenceType.external
            so_path = os.environ['PX_EXTINF']
            assert os.path.exists(so_path)
            self.gpu_id = 'gpu'

    def get_meta_info(self):
        info = {
            'platform': self.gpu_id.upper(), # more details?
            'software': 'PXPY, Python, C++'
        }
        return info

    def get_info(self):
        cfg = {
            'config_id': '_'.join([self.config['Type'], self.config['Inference'], self.gpu_id]),
            'uses_gpu': self.gpu_id == 'gpu',
            # some more custom fields
            'predict_iters': self.predict_iters,
            'train_iters': self.train_iters,
            'train_lbp_iters': self.infer_iters,
            'modeltype': self.config['Type'],
            'inferencetype': self.inference_str
        }
        return cfg

    # reliability check implementations

    def compute_pairwise_marginals(self, weights, structure, states):
        graph = px.create_graph(structure)

        # somehow this does not work, so use dummy samples
        # model = px.create_model(weights, graph, states.astype(np.uint8))
        dummy_samples = np.full((2, graph.nodes), np.max(states) - 1).astype(np.uint16)
        model = px.train(dummy_samples, graph=graph, iters=5, shared_states=True)
        np.copyto(model.weights, weights)
        
        marginals, _ = model.infer(inference=self.inference)
        _, pairwise = split_marginals(model, marginals)

        model.delete()
        graph.delete()

        return pairwise
