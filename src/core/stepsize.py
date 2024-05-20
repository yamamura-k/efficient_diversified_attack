import math
import torch

from utils import setup_logger

logger = setup_logger(__name__)


class Stepsize:
    """Stepsize update

    Attributes
    ----------
    epsilon : float
        Radius of the feasible region
    initial_stepsize : float
        initial stepsize \eta^0
    max_iter : int
        Maximum number of search iteration
    device : torch.device
        cpu / cuda
    strategy : str
        Stepsize controll method
    linesearch : Linesearh
    """

    def __init__(
        self,
        epsilon: float,
        initial_stepsize: float,
        max_iter: int,
        strategy: str,
        device: torch.device,
        *args,
        **kwargs,
    ):
        self.epsilon = epsilon
        self.initial_stepsize = initial_stepsize
        self.max_iter = max_iter
        self.device = device
        self.strategy = strategy

    def __call__(
        self,
        iteration: int,
        eta: torch.Tensor,
        inds: torch.Tensor,
        v,
        v_best=None,
        *args,
        **kwargs,
    ):
        bs = eta.shape[0]
        logger.debug(f"[ step strategy ] : {self.strategy}")
        if iteration < 0 and self.strategy != "line":
            return eta

        elif self.strategy == "static":
            return eta

        elif self.strategy == "apgd":
            eta[inds] /= 2
            v.updateVariable(v_best, inds)
            return eta

        elif self.strategy == "cos":
            max_stepsize = 2 * self.epsilon
            min_stepsize = 0.0
            eta = (
                min_stepsize
                + (max_stepsize - min_stepsize) * (1 + math.cos(math.pi * iteration / (self.max_iter + 1))) / 2
            )
            eta = eta * torch.ones(size=(bs, 1, 1, 1), device=self.device)
            return eta

        elif self.strategy == "cos_update":
            max_stepsize = 2 * self.epsilon
            min_stepsize = 0.0
            eta = (
                min_stepsize
                + (max_stepsize - min_stepsize) * (1 + math.cos(math.pi * iteration / (self.max_iter + 1))) / 2
            )
            # if iteration > 70:
            if iteration == int(self.max_iter * 0.65):
                # if iteration == 41:
                v.updateVariable(v_best, inds)
                # updateVariable(torch.ones((v.xk.shape[0], ), dtype=torch.bool), v, v_best)
            eta = eta * torch.ones(size=(bs, 1, 1, 1), device=self.device)
            return eta

        elif "exp-cosh" in self.strategy:
            p = float(self.strategy.split("-")[-1])
            X = 2 * p * iteration / self.max_iter - p
            eta = self.initial_stepsize * (1 - math.exp(X) / math.cosh(X) / 2)
            eta = eta * torch.ones(size=(bs, 1, 1, 1), device=self.device)
            return eta

        elif self.strategy == "exp":
            eta = self.initial_stepsize * math.exp(-pow(2 * iteration / self.max_iter, 2))
            eta = eta * torch.ones(size=(bs, 1, 1, 1), device=self.device)
            return eta

        elif self.strategy == "tanh":
            X = 2 * (iteration - 1) / self.max_iter
            eta = self.initial_stepsize * (math.tanh(X) - X + 1)
            eta = eta * torch.ones(size=(bs, 1, 1, 1), device=self.device)
            return eta

        elif self.strategy == "sin":
            X = iteration - self.max_iter
            eta = self.initial_stepsize * (math.sin(X) - X) / self.max_iter
            eta = eta * torch.ones(size=(bs, 1, 1, 1), device=self.device)
            return eta

        else:
            raise NotImplementedError(f"stepsize strategy [ {self.strategy} ] is not implemented.")
