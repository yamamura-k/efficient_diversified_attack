from itertools import combinations
import torch
import numpy as np

from tqdm import tqdm


from evaluator_base import EvaluatorBase
from metrics import DiversityIndex
from extension.count_target_classes import _get_target_classes
from utils import correct_param


class ADS(EvaluatorBase):
    def __init__(self, config, attacker, criterion, *args, **kwargs):
        super(ADS, self).__init__(config, *args, **kwargs)
        self.attacker = attacker
        self.criterion = criterion
        self.di = DiversityIndex(epsilon=config.param.epsilon, num_nodes=config.param.num_nodes)
        self.config = config.copy()
        self.in_indicator = config.in_indicator
        self.out_indicator = config.out_indicator

    @torch.no_grad()
    def __call__(
        self,
        x_best,
        x_test,
        y_test,
        acc,
        target_indices,
        target_image_indices_all,
        best_cw_loss_all,
        candidates,
        max_iter,
        stepsize,
        initial_point,
        bs,
        device,
        K=50,
        sample_ratio=0.01,
        n_algorithms=5,
        seed=0,
        strategy_idx=0,
    ):
        n_forward = 0
        n_backward = 0
        if strategy_idx == -1:
            import random
            random.seed(seed)
            settings = random.sample(candidates, n_algorithms)
            return (
                settings,
                n_forward,
                n_backward,
            )
        # take a sample set from all images
        self.x_advs_all = x_test.clone()
        self.target_indices = target_indices
        self.target_image_indices_all = target_image_indices_all
        self.best_cw_loss_all = best_cw_loss_all

        acc_subset = acc.clone()
        if sample_ratio > 1:
            n_test = sample_ratio
        else:
            n_test = int(sample_ratio * acc.sum().item())

        mask = acc_subset[acc].clone()

        if seed < 0:
            mask[n_test:] = False
        else:
            np.random.seed(seed)
            false_indices = torch.from_numpy(
                np.random.choice(np.arange(len(mask.squeeze())), size=n_test, replace=False)
            )
            mask[false_indices] = False
            mask = torch.logical_not(mask)
        acc_subset[acc] = mask.clone()

        search_points = []
        best_cw_loss = []
        target_classes = []

        param = self.config.param.copy()
        param_normalization = self.config.normalization.copy()
        param_stepsize = self.config.stepsize.copy()
        param_algorithm = self.config.algorithm.copy()
        param_initialpoint = self.config.initialpoint.copy()

        # update parameters of attacker
        param_normalization.norm_type = "sign"
        param_stepsize.strategy = "static"
        param.max_iter = max_iter
        param_stepsize.initial_stepsize = stepsize
        param_initialpoint.method = initial_point
        correct_param(param, param_initialpoint, param_stepsize, self.config.dataset)
        self.attacker.updateParameters(
            **param,
            param_algorithm=param_algorithm,
            param_normalization=param_normalization,
            param_initialpoint=param_initialpoint,
            param_stepsize=param_stepsize,
            device=device,
        )

        for (algo_name, criterion_name) in candidates:
            solution, n_forward, n_backward, accuracy = self.step(
                x_best=x_best,
                x_test=x_test,
                y_test=y_test,
                acc=acc_subset,
                max_iter=max_iter,
                bs=bs,
                algo_name=algo_name,
                criterion_name=criterion_name,
                device=device,
                n_forward=n_forward,
                n_backward=n_backward,
                y_target=None,
            )
            search_points.append(solution.x_adv[acc_subset].clone().cpu())
            best_cw_loss.append(solution.best_cw_loss[acc_subset].squeeze().clone().cpu())
            target_classes.append(solution.target_class[acc_subset].cpu().clone())

        search_points = torch.stack(search_points)
        best_cw_loss = torch.stack(best_cw_loss)
        target_classes = torch.stack(target_classes)
        DIs = []
        best_cw_loss_comb = []
        settings_comb = list(combinations(range(len(candidates)), n_algorithms))
        for v in tqdm(settings_comb):
            criterions = set([candidates[ind][1] for ind in v])
            if strategy_idx < 5 and len(criterions) < len(v):
                DIs.append(0.0)
                best_cw_loss_comb.append(-100.0)
                continue
            else:
                _best_loss = best_cw_loss[v, :, -1].max(0).values
                # diversity in the output space
                if self.out_indicator == "classes":
                    target_class = target_classes[v, :, :].clone()
                    out_indicator = self.get_target_classes(target_class=target_class)
                else:
                    out_indicator = torch.ones((target_classes.shape[1]))
                # diversity in the input space
                if self.in_indicator == "di":
                    X = search_points[v, :, :, :].clone().to(device)
                    # weights = torch.exp(-_best_loss).numpy()
                    weights = None
                    in_indicator = self.di.get(X, weights=weights)
                elif self.in_indicator == "median":
                    _bs = search_points.shape[1]
                    X = (
                        search_points[v, :, :, :]
                        .clone()
                        .to(device)
                        .permute(1, 0, 2, 3, 4)
                        .reshape((_bs, n_algorithms, -1))
                    )
                    X_mean = X.mean(dim=1, keepdim=True)
                    in_indicator = torch.cdist(X, X_mean).mean(dim=1).squeeze().cpu()
                else:
                    in_indicator = torch.ones((target_classes.shape[1]))
                DIs.append((in_indicator * out_indicator).mean().item())
                best_cw_loss_comb.append(_best_loss.mean().item())

        if strategy_idx == 0:
            inds_diversity = torch.argsort(torch.tensor(DIs))
            inds_loss_values = torch.argsort(torch.tensor(best_cw_loss_comb)[inds_diversity[-K:]])
            inds = inds_diversity[:-K][inds_loss_values[-1]]
        elif strategy_idx == 1:
            inds_loss_values = torch.argsort(torch.tensor(best_cw_loss_comb))
            inds_diversity = torch.argsort(torch.tensor(DIs)[inds_loss_values[-K:]])
            inds = inds_loss_values[:-K][inds_diversity[-1]]
        elif strategy_idx == 2: # default setting
            inds = torch.argsort(torch.tensor(DIs))[-1]
        elif strategy_idx == 3:
            inds = torch.argsort(torch.tensor(best_cw_loss_comb))[-1]
        elif strategy_idx == 4:
            inds = torch.argsort(torch.tensor(DIs) * torch.tensor(best_cw_loss_comb))[-1]
        elif strategy_idx == 5: # R-ADS
            inds = torch.argsort(torch.tensor(DIs))[0]
        else:
            raise NotImplementedError

        settings = [candidates[i] for i in settings_comb[inds]]
        return (
            settings,
            n_forward,
            n_backward,
        )

    def get_target_classes(self, target_class):
        _n_classes = _get_target_classes(target_class.numpy())
        n_classes = torch.from_numpy(_n_classes)
        return n_classes
