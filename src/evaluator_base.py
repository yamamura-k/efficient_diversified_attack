import math
import torch


from utils import setup_logger, InformationWriter
from core.container import Information

logger = setup_logger(__name__)


class EvaluatorBase:
    """Management of the input images and output of the attack

    Attributes
    ----------
    config : dict
        Config of the evaluation
    postprocess : InformationWriter
        Export the search information
    x_test : torch.Tensor
        Natural images
    y_test : torch.Tensor
        Ground truth label for x_test
    """

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.postprocess = InformationWriter()
        self.x_test = None
        self.y_test = None

    @torch.no_grad()
    def step(
        self,
        x_best,
        x_test,
        y_test,
        acc,
        max_iter,
        bs,
        algo_name,
        criterion_name,
        device,
        n_forward,
        n_backward,
        y_target=None,
        *args,
        **kwargs,
    ):
        """Conduct an adversarial attack

        Parameters
        ----------
        x_best : torch.Tensor / None
            Incumbent solution
        x_test : torch.Tensor
            Natural image
        y_test : torch.Tensor
            Ground truth label of x_test
        acc : torch.Tensor
            Bool tensor to filter the image
        max_iter : int
            Maximum number of search iterations
        bs : int
            Batch size
        algo_name : str
            Name of algorithm used in the attack
        criterion_name : str
            Name of the objective function
        device : torch.device
            cpu / cuda
        n_forward : int
            Number of forward propagation
        n_backward : in
            Number of backward propagation
        y_target : torch.Tensor, optional
            Targe class label for misclassification, by default None

        Returns
        -------
        solution : Information
            Search information
        n_forward : int
            Number of forward propagation
        n_backward : int
            Number of backward propagation
        """
        logger.info("[ Step ]")

        minus_ones = torch.full((len(x_test), max_iter + 1), -1.0)
        minus_ones_int = torch.full((len(x_test), max_iter + 1), -1, dtype=torch.int)
        __max_iter = (
            max_iter
            + 1
            - self.config.initialpoint.odi_iter
            * int(
                self.attacker.initialpoint.method in {"odi", "opt-based", "pas", "random-odi", "random-pas"}
            )
        )
        solution = Information(
            best_loss=minus_ones.clone(),
            best_cw_loss=minus_ones.clone(),
            best_softmax_cw_loss=minus_ones.clone(),
            current_loss=minus_ones.clone(),
            current_cw_loss=minus_ones.clone(),
            current_softmax_cw_loss=minus_ones.clone(),
            step_size=minus_ones.clone() if self.postprocess.export_level <= 40 else None,
            diversity_index_1=minus_ones.clone() if self.postprocess.export_level <= 40 else None,
            diversity_index_2=minus_ones.clone() if self.postprocess.export_level <= 40 else None,
            target_class=minus_ones_int.clone(),
            n_projected_elms=minus_ones_int.clone() if self.postprocess.export_level <= 30 else None,
            n_boundary_elms=minus_ones_int.clone() if self.postprocess.export_level <= 30 else None,
            delta_x=minus_ones.clone() if self.postprocess.export_level <= 20 else None,
            grad_norm=minus_ones.clone() if self.postprocess.export_level <= 20 else None,
            x_adv=x_test.clone(),
            x_advs=torch.zeros((*x_test.shape, __max_iter)) if self.postprocess.export_level <= 10 else None,
            grad_adv=torch.zeros_like(x_test) if self.postprocess.export_level <= 10 else None,
        )
        x, y = x_test.clone(), y_test.clone()
        n_examples = acc[self.target_indices].sum().item()
        nbatches = math.ceil(n_examples / bs)
        n_success = 0
        logger.info(f"idx {-1}: ASR = {n_success} / 0")
        _accuracy = acc.clone()

        for idx in range(nbatches):
            begin = idx * bs
            end = min((idx + 1) * bs, n_examples)

            target_image_indices = self.target_image_indices_all[self.target_indices][acc[self.target_indices]][
                begin:end
            ]

            if len(x[target_image_indices]) > 0:
                if isinstance(x_best, torch.Tensor):
                    x_best_batch = x_best[target_image_indices].clone()
                else:
                    x_best_batch = None
                if isinstance(y_target, torch.Tensor):
                    y_target_batch = y_target[target_image_indices].clone().to(device)
                else:
                    y_target_batch = None
                solution_batch, n_forward, n_backward, accuracy = self.attacker.attack(
                    x[target_image_indices].clone().to(device),
                    y[target_image_indices].clone().to(device),
                    self.criterion,
                    criterion_name,
                    algo_name,
                    x_best=x_best_batch,
                    y_target=y_target_batch,
                    n_forward=n_forward,
                    n_backward=n_backward,
                    *args,
                    **kwargs,
                )
                assert solution_batch.target_class is not None
                if solution_batch["x_advs"] is not None:
                    solution_batch["x_advs"] = solution_batch["x_advs"].permute(1, 2, 3, 4, 0)
                _accuracy[target_image_indices] = accuracy.clone().cpu()
                solution.updateInformation(solution_batch, target_image_indices)
                n_success += torch.logical_not(accuracy).sum().item()
                logger.info(f"idx {idx}: ASR = {n_success} / {end}")

                # update global best cw loss values
                if (self.best_cw_loss_all[target_image_indices] == -1).all():
                    self.best_cw_loss_all[target_image_indices] = solution_batch.best_cw_loss[:, -1].clone()
                    self.x_advs_all[target_image_indices] = solution_batch.x_adv.clone()
                else:
                    ind = self.best_cw_loss_all[target_image_indices] < solution_batch.best_cw_loss[:, -1]
                    _tmp = self.best_cw_loss_all[target_image_indices].clone()
                    _tmp[ind] = solution_batch.best_cw_loss[:, -1][ind].clone()
                    self.best_cw_loss_all[target_image_indices] = _tmp.clone()
                    x_tmp = self.x_advs_all[target_image_indices].clone()
                    ind_2 = torch.logical_or(torch.logical_not(accuracy.cpu()), ind)
                    x_tmp[ind_2] = solution_batch.x_adv[ind_2].clone()
                    self.x_advs_all[target_image_indices] = x_tmp.clone()
            else:
                logger.warning(f"#target image = 0 at batch {idx}.")

        return solution, n_forward, n_backward, _accuracy

    def run(
        self,
        model,
        x_test,
        y_test,
        target_image_indices_all,
        target_indices,
        EXPORT_LEVEL=60,
        EXPERIMENT=False,
    ):
        raise NotImplementedError
