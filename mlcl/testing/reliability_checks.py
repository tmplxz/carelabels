import os
import pickle

import numpy as np

from mlcl.util import check_complexity, aggregate_memory, complexity_classes, KL


def runtime_complexity(implementation, args):

    varied = ['vertices', 'states'] # TODO parametrize via logs
    measurements = list(args['measurements'].values())
    expected = args['theoretical']

    # rating depends on the highest complexity of all varied factors O notation
    complexity = check_complexity(measurements, varied, 'APPLY_RUNTIME_WCT')

    details = [f'Theoretical runtime complexity: {expected} ({complexity_classes[expected]})'] +\
        [f'Runtime complexity with scaling "{vary}": {cplx} ({complexity_classes[cplx]})' for vary, cplx in complexity.items() if not np.isnan(cplx)]

    return {
        'success': max(complexity.values()) <= expected,
        'description': 'The theoretical runtime complexity should align with the asymptotic runtime behaviour on scaling data.',
        'details': '\n'.join(details)
    }


def memory_complexity(implementation, args):

    varied = ['vertices', 'states'] # TODO parametrize via logs
    measurements = list(args['measurements'].values())
    for meas in measurements:
        meas['APPLY_MEMORY'] = aggregate_memory(meas, rep_agg=None) + aggregate_memory(meas, gpu=True, rep_agg=None)
    expected = args['theoretical']

    # rating depends on the highest complexity of all varied factors O notation
    complexity = check_complexity(measurements, varied, 'APPLY_MEMORY')

    details = [f'Theoretical memory complexity: {expected} ({complexity_classes[expected]})'] +\
        [f'Memory complexity with scaling "{vary}": {cplx} ({complexity_classes[cplx]})' for vary, cplx in complexity.items() if not np.isnan(cplx)]

    return {
        'success': max(complexity.values()) <= expected,
        'description': 'The theoretical memory complexity should align with the asymptotic memory behaviour on scaling data.',
        'details': '\n'.join(details)
    }


def load_mrf_reliability_data():
    ds = ['chain_nodes20_states20_nsamples50000.pkl', 'grid_nodes5_states5_nsamples50000.pkl']
    dirn = os.path.dirname(os.path.realpath(__file__))
    loaded_ds = []
    for d in ds:
        with open(os.path.join(dirn, 'datasets', d), 'rb') as fd:
            content = pickle.load(fd)
            content['name'] = d
            loaded_ds.append(content)
    return loaded_ds


def probability_recovery(implementation, args):

    threshold = 1e-3
    kl_diffs = []
    names = []
    data = load_mrf_reliability_data()

    for ds in data:

        states = ds['states']
        edgelist = ds['edgelist']
        weights = ds['weights']

        mu_star = ds['mu']
        mu_hat = implementation.compute_pairwise_marginals(weights, edgelist, states)

        offsets = []
        off = 0
        for (s, t) in edgelist:
            off += int(states[s] * states[t])
            offsets.append(off)

        mu_star_cliques = np.split(mu_star, offsets)[:-1]
        mu_hat_cliques = np.split(mu_hat, offsets)[:-1]
        kls = [KL(c1, c2) for (c1, c2) in zip(mu_star_cliques, mu_hat_cliques)]

        kl_diffs.append(np.max(kls))
        names.append(ds['name'])

    return {
        'success': all([kl < threshold for kl in kl_diffs]),
        'description': 'The error on the recovered marginal vector has to be below a threshold defined by the Hoeffding Bound.',
        'details': '\n'.join([f'{name} - Maximum KL divergence {kl}' for name, kl in zip(names, kl_diffs)])
    }
