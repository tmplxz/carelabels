import os

import torch
import numpy as np
import pandas as pd

from .base import RobustnessTest
from mlcl.torch_utils import AverageMeter


IMAGE_NET_C = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'speckle_noise',
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
]


def evaluate_error(df):
    df['top1_error'] = 100 - df['top1']

    # Compute Corruption Error for each severity level and each model
    corruption_df = df.groupby(['model.name', 'corruption'], as_index=False).sum()

    # Remove irrelevant column
    corruption_df.drop(columns=['severity'], inplace=True)

    corruption_df['top1_error_normalized'] = 0.0

    # Normalize with imagenet!
    for index, row in corruption_df.iterrows():
        row = row.copy()
        ref_key = 'alexnet'
        corruption = row['corruption']

        # Get row which is used as denominator for the normalization
        ref_row = corruption_df[(corruption_df['model.name'] == ref_key) & (corruption_df['corruption'] == corruption)]

        for c in ['top1_error']:
            # Skip meta information
            if c in ['model.name', 'corruption']:
                continue
            # Apply "normalization"

            # Error for model f, corruption c summed for all severity levels.
            E_f_s_c = float(row[c])
            E_alexnet_s_c = float(ref_row[c])

            CE_f_c = E_f_s_c / E_alexnet_s_c
            corruption_df.loc[index, f'{c}_normalized'] = CE_f_c

    # Compute Mean Corruption Error
    mean_corruption_df = corruption_df.groupby(['model.name'], as_index=False).mean()
    return mean_corruption_df


class CorruptionTest(RobustnessTest):


    def __init__(self, implementation, benchmark, logdir):
        super().__init__(implementation, benchmark, logdir)
        self.model = implementation.model
        self.model.eval()
        self.model.to(self.implementation.device)
        self.severity_levels = [1, 2, 3, 4, 5]
        self.corruptions = IMAGE_NET_C
        self.full_log = os.path.join(os.path.dirname(self.logname), os.path.basename(self.logname).split('.')[0] + '.csv')


    def run_test(self):

        data_handler = self.implementation.ds_class(self.benchmark)

        if os.path.exists(self.full_log):
            result_df = pd.read_csv(self.full_log)
        else:
            result_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'corruption_reference.csv'))
            result_df = result_df.drop(columns='top5')

            for corruption in self.corruptions:
                for severity in self.severity_levels:

                    avg_acc = AverageMeter('top1')
                    data = data_handler.get_corruption(corruption, severity)
                    if data is None:
                        return {
                            'score': 0,
                            'description': 'Error during running the corruption robustness tests, could not find data',
                            'details': {
                                'top1': 0,
                                'top1_error': 0,
                                'top1_error_normalized': 0
                            }
                        }

                    for x, y in data:
                        x, y = x.to(self.implementation.device), y.to(self.implementation.device)
                        with torch.no_grad():
                            y_hat = self.model(x).max(1)[1]
                        avg_acc.update(np.count_nonzero(y_hat==y) / y_hat.shape[0] * 100, y_hat.shape[0])

                    result_df = result_df.append({
                        'model.name': 'tested',
                        'corruption': corruption,
                        'severity': severity,
                        'top1': avg_acc.avg,
                    }, ignore_index=True)

            result_df.to_csv(self.full_log)

        error_df = evaluate_error(result_df)
        
        return {
            'score': error_df[error_df['model.name'] == 'tested']['top1_error_normalized'].item(),
            'description': 'Common visual image corruptions with different levels of severity.',
            'details': {
                'top1': error_df[error_df['model.name'] == 'tested']['top1'].item(),
                'top1_error': error_df[error_df['model.name'] == 'tested']['top1_error'].item(),
                'top1_error_normalized': error_df[error_df['model.name'] == 'tested']['top1_error_normalized'].item()
            }
        }
