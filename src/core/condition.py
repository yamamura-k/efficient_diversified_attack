import gc
import torch

from core.base import Condition
from utils import setup_logger, DEBUG

logger = setup_logger(__name__)


class StepsizeCondition(Condition):
    """Compute the conditions for updating the step size

    Attributes
    ----------
    max_iter : int
        Maximum iteration
    rho : float
        Threshould of the APGD's step size rule
    w : int
        Checkpoint of the APGD's step size rule
    size_decr : int
        Checkpoint of the APGD's step size rule
    n_iter_min : int
        Checkpoint of the APGD's step size rule
    use_cw_value : bool
        If True, use CW loss value for APGD's step size update
    checkpoint : int
        Checkpoint of the APGD's step size rule
    checkpoint_prev : int
        Previous checkpoint of the APGD's step size rule
    device : torch.device
        Tensor location
    reduced_last_check : torch.Tensor
        Tensor for APGD's update rule
    best_loss_last_check : torch.Tensor
        Tensor for APGD's update rule
    """

    def __init__(
        self,
        max_iter: int,
        device: str,
        use_cw_value: bool = False,
        rho: float = 0.75,
        w_ratio: float = 0.22,
        size_decr_ratio: float = 0.03,
        n_iter_min_ratio: float = 0.06,
        *args,
        **kwargs,
    ):
        super(StepsizeCondition, self).__init__()
        self.max_iter = max_iter
        self.rho = rho
        self.w_ratio = w_ratio
        self.size_decr = size_decr_ratio
        self.n_iter_min_ratio = n_iter_min_ratio
        self.w = int(w_ratio * max_iter)
        self.size_decr = int(size_decr_ratio * max_iter)
        self.n_iter_min = int(n_iter_min_ratio * max_iter)
        self.use_cw_value = use_cw_value
        # constant values
        self.checkpoint = self.w
        self.checkpoint_prev = 0
        self.device = device
        self.reduced_last_check = None
        self.best_loss_last_check = None

    @torch.no_grad()
    def __call__(self, l, iteration: int, name="apgd", v=None):
        if name == "static":
            return torch.zeros_like(l.current_loss[:, 0], device=self.device, dtype=bool)
        elif name == "cos":
            return None
        elif name == "apgd":
            if iteration + 1 == self.checkpoint:
                if self.reduced_last_check is None:
                    self.reduced_last_check = torch.ones(size=(l.current_cw_loss.shape[0],), dtype=bool)
                if self.best_loss_last_check is None:
                    self.best_loss_last_check = (
                        l.best_cw_loss[:, 0].clone() if self.use_cw_value else l.best_loss[:, 0].clone()
                    )
                assert (
                    self.checkpoint - self.checkpoint_prev == self.w
                ), f"Assertion failed: illigal checkpoint at iteration {iteration}."
                # How many times the best objective is updated?
                if self.use_cw_value:
                    num_updates = self.check_oscillation(iteration, l.current_cw_loss)

                else:
                    num_updates = self.check_oscillation(iteration, l.current_loss)
                condition1 = self.rho * self.w * torch.ones_like(num_updates) >= num_updates
                # Whether the best objective is updated or not
                best_loss = (
                    l.best_cw_loss[:, 1:][:, iteration].clone()
                    if self.use_cw_value
                    else l.best_loss[:, 1:][:, iteration].clone()
                )
                condition2 = self.best_loss_last_check >= best_loss
                condition2 = torch.logical_and(condition2, torch.logical_not(self.reduced_last_check))
                condition = torch.logical_or(condition1, condition2)
                # Update the checkpoints
                self.checkpoint_prev = self.checkpoint
                self.w = max(self.w - self.size_decr, self.n_iter_min)
                self.checkpoint += self.w
                self.reduced_last_check = condition.clone()
                self.best_loss_last_check = best_loss.clone()
                if DEBUG:
                    return condition.to(self.device), condition1, condition2
                else:
                    return condition.to(self.device)
            else:
                return torch.zeros_like(l.current_loss[:, 0], device=self.device, dtype=bool)
        else:
            return None

    def reset(self):
        del self.reduced_last_check, self.best_loss_last_check
        gc.collect()
        self.w = int(self.w_ratio * self.max_iter)
        self.checkpoint = self.w
        self.checkpoint_prev = 0
        self.reduced_last_check = None
        self.best_loss_last_check = None

    @torch.inference_mode()
    def check_oscillation(self, iteration, loss_steps):
        num_updates = torch.zeros(loss_steps.shape[0])
        for counter5 in range(self.w):
            num_updates += loss_steps[:, 1:][:, iteration - counter5] > loss_steps[:, 1:][:, iteration - counter5 - 1]
        return num_updates


class updateOnce(Condition):
    def __init__(self, bs: int, device: torch.device):
        self.not_updated = torch.ones(size=(bs,), device=device)

    @torch.no_grad()
    def __call__(self, inds: torch.Tensor, v1, v2):
        v1.updateVariable(v2, torch.logical_and(self.not_updated, inds))
        self.not_updated[inds] = False

    def reset(self):
        self.not_updated = torch.ones_like(self.not_updated)
