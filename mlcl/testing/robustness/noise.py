from easydict import EasyDict

import numpy as np
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
import torch

from .base import RobustnessTest


class NoiseTest(RobustnessTest):

    
    def __init__(self, implementation, benchmark, logdir):
        super().__init__(implementation, benchmark, logdir)
        self.model = implementation.model
        self.model.eval()
        self.model.to(self.implementation.device)
        # additional hyperparameters
        self.eps = 0.001
        self.eps_iter = 0.001
        self.norm = np.inf


    def estimate_safe_ball(self, val_loader, threshold=1.0e-4):
        """
        Estimate the radius of the safe ball around the data points
        """
        found_eps = False
        eps_tmp = self.eps
        eps_iter_tmp = self.eps_iter
        while not found_eps:

            report = EasyDict(
                nb_test=0.0,
                correct=0.0,
                correct_eps_inf=0.0,
                drop_acc_inf=0.0,
                top5_inf=0.0,
            )
            for x, y in val_loader:

                x, y = x.to(self.implementation.device), y.to(self.implementation.device)

                _, y_pred = self.model(x).max(1) # model prediction on clean examples
                correct = y_pred.eq(y).sum().item()

                report.nb_test += y.size(0)
                report.correct += correct

                params = {
                    "model_fn": self.model,
                    "x": x,
                    "eps": eps_tmp, # TODO shouldn't this be eps_tmp?
                    "norm": self.norm,
                    "nb_iter": 10,
                    "eps_iter": eps_iter_tmp,
                }

                x_pgd = projected_gradient_descent(**params).to(self.implementation.device)
                with torch.no_grad():
                    y_pgd = self.model(x_pgd)
                
                _, y_pgd = y_pgd.max(1)

                report['correct_eps_inf'] += float(y_pgd.eq(y).sum().item())
                report['drop_acc_inf'] += (float(correct) - float(y_pgd.eq(y).sum().item()))

            if (report.drop_acc_inf / report.nb_test * 100.0) < threshold:
                found_eps = True
            else:
                if eps_tmp != self.eps:
                    found_eps = True
                eps_tmp /= 2.0
                eps_iter_tmp /= 2.0
        return eps_tmp, report


    def run_test(self):

        data_handler = self.implementation.ds_class(self.benchmark)
        data = data_handler.get_apply()

        estimated_eps, report = self.estimate_safe_ball(data)
        
        # also return details?
        clean = report.correct / report.nb_test * 100.0,
        pgd_norm_inf = report.correct_eps_inf / report.nb_test * 100.0,
        # pgd_norm_inf_top_5'.format(args.eps): report.top5_inf / report.nb_test * 100.0,
        pgd_drop_acc_inf = report.drop_acc_inf / report.nb_test * 100.0,

        return {
            'score': estimated_eps,
            'description': 'Default',
            'details': {
                'clean': clean,
                'pgd_norm_inf': pgd_norm_inf,
                'pgd_drop_acc_inf': pgd_drop_acc_inf
            }
        }