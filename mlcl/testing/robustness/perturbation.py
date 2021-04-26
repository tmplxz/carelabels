import os

import torch
import numpy as np
import pandas as pd

from .base import RobustnessTest


IMAGE_NET_P = [
    'brightness',
    'gaussian_noise',
    'motion_blur',
    'rotate',
    'scale',
    'shot_noise',
    'snow',
    'tilt',
    'translate',
    'zoom_blur',
]


def flip_counts(seq, diff=1):
    """
    sum_j=2: 1(x_j-1 != x_j)
    Parameters
    ----------
    seq :
    diff:

    Returns
    -------

    """
    idx = np.arange(seq.shape[0] * (seq.shape[1] - diff))
    u_idx = np.unravel_index(idx, (seq.shape[0], seq.shape[1] - diff))

    # mask = np.zeros(seq.size - diff).astype(np.bool)
    # for i in range(1, diff + 1):
    # mask = np.logical_or(mask, seq[idx] != seq[idx + i])
    return np.where(seq[u_idx] != seq[u_idx[0], u_idx[1] + 1])[0].size


def evaluate_error(perturbation_df):

    perturbation_df['flip_rate'] = 0.0

    # Normalize with imagenet!
    for index, row in perturbation_df.iterrows():
        row = row.copy()
        ref_key = 'alexnet'
        perturbation = row['perturbation']

        ref_row = perturbation_df[
            (perturbation_df['model.name'] == ref_key) & (perturbation_df['perturbation'] == perturbation)]

        fp_fp = float(row['flip_probability'])
        fp_alexnet_p = float(ref_row['flip_probability'])

        val = fp_fp / fp_alexnet_p

        perturbation_df.loc[index, 'flip_rate'] = val

    mean_flip_rate = perturbation_df.groupby(['model.name'], as_index=False).mean()
    mean_flip_rate.rename(columns={'flip_probability': 'mean_flip_probability',
                                   'flip_rate': 'mean_flip_rate'}, inplace=True)

    return mean_flip_rate


class PerturbationTest(RobustnessTest):

    def __init__(self, implementation, benchmark, logdir):
        super().__init__(implementation, benchmark, logdir)
        self.model = implementation.model
        self.model.eval()
        self.model.to(self.implementation.device)
        self.perturbations = IMAGE_NET_P
        self.full_log = os.path.join(os.path.dirname(self.logname), os.path.basename(self.logname).split('.')[0] + '.csv')
    
    def run_test(self):

        data_handler = self.implementation.ds_class(self.benchmark)

        if os.path.exists(self.full_log):
            result_df = pd.read_csv(self.full_log)
        else:
            result_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'perturbation_reference.csv'))

            for perturbation in self.perturbations:

                data = data_handler.get_perturbation(perturbation)
                if data is None:
                    return {
                        'score': 0,
                        'description': 'Error during running the perturbation robustness tests, could not find data',
                        'details': {
                            'top1': 0,
                            'top1_error': 0,
                            'top1_error_normalized': 0
                        }
                    }

                diff = 1
                flip_count_total = 0

                for x, y in data:
                    x, y = x.to(self.implementation.device), y.to(self.implementation.device)

                    n_videos = x.size(0)
                    video_len = x.size(1)

                    # Inference
                    with torch.no_grad():
                        x = x.view(-1, 3, 224, 224)

                        y_hat_frame_batches = self.model(x)

                        preds = torch.reshape(y_hat_frame_batches.argmax(axis=1), (n_videos, video_len))
                        fp_cnt = flip_counts(preds.cpu().detach().numpy())

                        flip_count_total += fp_cnt

                flip_probability = flip_count_total / (len(data) * (video_len - diff))

                result_df = result_df.append({
                    'model.name': 'tested',
                    'perturbation': perturbation,
                    'flip_count': flip_count_total,
                    'flip_probability': flip_probability
                }, ignore_index=True)

            result_df.to_csv(self.full_log)

        error_df = evaluate_error(result_df)
        
        return {
            'score': error_df[error_df['model.name'] == 'tested']['mean_flip_probability'].item(),
            'description': 'Sequences of perturbed or subtly changed images.',
            'details': {
                'flip_count': error_df[error_df['model.name'] == 'tested']['flip_count'].item(),
                'mean_flip_probability': error_df[error_df['model.name'] == 'tested']['mean_flip_probability'].item(),
                'mean_flip_rate': error_df[error_df['model.name'] == 'tested']['mean_flip_rate'].item()
            }
        }
