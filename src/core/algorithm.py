import torch
from math import sqrt, log
from functools import lru_cache

from utils import setup_logger
from core.normalization import Normalization

logger = setup_logger(__name__)


@lru_cache(maxsize=500)
def updateRho(rho):
    return (1 + sqrt(1 + 4 * pow(rho, 2))) / 2


@lru_cache(maxsize=500)
def updateGamma(rho, rho_1):
    return (rho_1 - 1) / rho


@lru_cache(maxsize=500)
def updateRho2(rho, q=0):
    return (q - pow(rho, 2) + sqrt(pow(q - rho, 2) + 4 * pow(rho, 2))) / 2


@lru_cache(maxsize=500)
def updateGamma2(rho, rho_1):
    return rho_1 * (1 - rho_1) / (rho + pow(rho_1, 2))


class Algorithm:
    """Compute next search point following the update fomulas.

    Atributes
    ---------
    normalize : Normalization
        Normalization method for the update direction.
    projection : Projection
        Projection onto the feasible region.
    momentum_alpha : float
        Coefficient of the momentum term.
    beta_method : str
        Calculation method for coefficient beta of CG method.
    mu : float
        Coefficient of the gradient momentum.
    rho_1 : float
        Used in Nesterov's update
    rho : float
        Used in Nesterov's update
    gamma : float
        Used in Nesterov's update
    """

    def __init__(self, momentum_alpha, beta_method, projection=None, *args, **kwargs):
        self.normalize = Normalization(*args, **kwargs)
        self.projection = projection
        self.momentum_alpha = momentum_alpha
        self.beta_method = beta_method
        self.mu = 1.0
        self.rho_1 = 1.0
        self.rho = 1.0
        self.gamma = 0.0

    @torch.no_grad()
    def __call__(
        self,
        v,
        eta,
        algo_name="GD",
        iteration=-1,
        indices=None,
        update_v=True,
        *args,
        **kwargs,
    ):
        """Compute next search point and update Variables.

        Parameters
        ----------
        v : Variable
            Search points and its gradients.
        eta : torch.Tensor
            Step size
        algo_name : str, optional
            Update method, by default "GD"
        iteration : int, optional
            Current iteration, by default -1
        indices : torch.Tensor, optional
            Indices to be updated, by default None
        update_v : bool, optional
            Whether update the variables or not, by default True

        Returns
        -------
        torch.Tensor
            The next search point.

        Raises
        ------
        NotImplementedError
            Error occurs if the given algo_name is not implemented.
        """
        x_nxt = None
        logger.debug(f"[ algorithm ]: {algo_name}")
        if algo_name == "GD":
            x_nxt = self.projection(v.xk + eta * self.normalize(v.gradk))

        elif algo_name == "MGD":
            # Implementation of Momentum-FGSM update
            if iteration == 0 or not hasattr(v, "grad_momentum"):
                v.grad_momentum = v.gradk.clone()
            else:
                v.grad_momentum += self.mu * v.gradk / v.gradk.norm(p=1, dim=(1, 2, 3), keepdim=True)
            x_nxt = self.projection(v.xk + eta * self.normalize(v.grad_momentum))

        elif algo_name == "MGD_l2":
            # Implementation of Momentum-FGSM update
            if iteration == 0 or not hasattr(v, "grad_momentum"):
                v.grad_momentum = v.gradk.clone()
            else:
                v.grad_momentum += self.mu * v.gradk / v.gradk.norm(p=2, dim=(1, 2, 3), keepdim=True)
            x_nxt = self.projection(v.xk + eta * self.normalize(v.grad_momentum))

        elif algo_name == "MGD_normal":
            if iteration == 0 or not hasattr(v, "grad_momentum"):
                v.grad_momentum = v.gradk.clone()
            else:
                v.grad_momentum += self.mu * v.gradk
            x_nxt = self.projection(v.xk + eta * self.normalize(v.grad_momentum))

        elif algo_name == "Nes":
            if iteration <= 0:
                self.reset()
            self.rho_1 = self.rho
            self.rho = updateRho2(self.rho)
            self.gamma = updateGamma2(self.rho, self.rho_1)
            v.x_nes = v.xk + self.gamma * (v.xk - v.xk_1)

        elif algo_name == "Nes_":
            if iteration <= 0:
                self.reset()
            self.rho_1 = self.rho
            self.rho = updateRho(self.rho)
            self.gamma = updateGamma(self.rho, self.rho_1)
            v.x_nes = v.xk + self.gamma * (v.xk - v.xk_1)

        elif algo_name == "Nes2":
            x_nxt = self.projection(v.x_nes + eta * self.normalize(v.grad_nes))

        elif algo_name == "CG":
            if iteration > 0:
                betak = self.getBeta(v, method=self.beta_method, use_clamp=False)
                v.sk = v.gradk + betak * v.sk_1
                logger.debug(f"beta({self.beta_method}) = {betak.mean().item():.4f}")
            x_nxt = self.projection(v.xk + eta * self.normalize(v.sk))

        elif algo_name == "APGD":
            grad_2 = (v.xk - v.xk_1).clone()
            zk = self.projection(v.xk + eta * self.normalize(v.gradk))
            momentum_alpha = 1.0 if iteration == 0 else self.momentum_alpha
            x_nxt = self.projection(v.xk + momentum_alpha * (zk - v.xk) + (1.0 - momentum_alpha) * grad_2)
        elif algo_name == "Langevin":
            c = 1
            eps_k = torch.rand(v.gradk.shape[0], device=v.gradk.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            d = eta * self.normalize(v.gradk) + (2 * c / log(2 + iteration * 100) * eta).sqrt() * eps_k
            # d = eta * v.gradk + (2 * c / log(2 + iteration) * eta).sqrt() * eps_k
            # d = eta * self.normalize(v.gradk + (2 * c / log(2 + 10000 * iteration) * eta).sqrt() * eps_k)
            # d = self.normalize(eta * v.gradk + (2 * c / log(2 + iteration) * eta).sqrt() * eps_k)
            x_nxt = self.projection(v.xk + d)
        else:
            raise NotImplementedError(f"{algo_name} is not implemented.")

        if update_v and x_nxt is not None:
            if indices is None:
                v.xk_1 = v.xk.clone()
                v.sk_1 = v.sk.clone()
                v.gradk_1 = v.gradk.clone()
                v.xk = x_nxt.clone()
            else:
                v.xk_1[indices] = v.xk[indices].clone()
                v.sk_1[indices] = v.sk[indices].clone()
                v.gradk_1[indices] = v.gradk[indices].clone()
                v.xk[indices] = x_nxt[indices].clone()
        if x_nxt is None:
            x_nxt = v.x_nes.clone()
        return x_nxt

    @torch.no_grad()
    def getBeta(self, v, method="HS", use_clamp=False):
        """Compute beta for CG method.

        Parameters
        ----------
        v : Variable
            Search points and its gradients.
        method : str, optional
            How to compute beta, by default "HS"
        use_clamp : bool, optional
            Clamp beta to [0, 1] if True, by default False

        Returns
        -------
        torch.Tensor
            beta

        Raises
        ------
        NotImplementedError
            if compute method is not implemented, raise an error.
        """
        bs = v.gradk.shape[0]
        _sk_1 = v.sk_1.reshape(bs, -1)
        _gradk = -v.gradk.reshape(bs, -1)
        _gradk_1 = -v.gradk_1.reshape(bs, -1)
        yk = _gradk - _gradk_1
        if method == "HS":
            betak = -(_gradk * yk).sum(dim=1) / (_sk_1 * yk).sum(dim=1)
        elif method == "FR":
            betak = v.gradk.norm(p=2, dim=(1, 2, 3)).pow(2) / v.gradk_1.norm(p=2, dim=(1, 2, 3)).pow(2)
        elif method == "PR":
            betak = (_gradk * yk).sum(dim=1) / v.gradk_1.norm(p=2, dim=(1, 2, 3)).pow(2)
        elif method == "DY":
            betak = v.gradk.norm(p=2, dim=(1, 2, 3)).pow(2) / (_sk_1 * yk).sum(dim=1)
        elif method == "HZ":
            betak = ((yk - 2 * _sk_1 * (yk.norm(p=2).pow(2) / (_sk_1 * yk).sum(dim=1)).unsqueeze(-1)) * _gradk).sum(
                dim=1
            ) / (_sk_1 * yk).sum(dim=1)
        elif method == "DL":
            betak = ((yk - _sk_1) * _gradk).sum(dim=1) / (_sk_1 * yk).sum(dim=1)
        elif method == "LS":
            betak = -(_gradk * yk).sum(dim=1) / (_sk_1 * _gradk_1).sum(dim=1)
        elif method == "RMIL":
            betak = (_gradk * yk).sum(dim=1) / v.sk_1.norm(p=2, dim=(1, 2, 3)).pow(2)
        elif method == "RMIL+":
            betak = (_gradk * (yk - _sk_1)).sum(dim=1) / v.sk_1.norm(p=2, dim=(1, 2, 3)).pow(2)
        else:
            raise NotImplementedError(f"{method} is not implemented.")

        if use_clamp:
            betak = betak.clamp(min=0)
        betak[torch.isnan(betak)] = 0.0
        return betak.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    def reset(self):
        """Reset rho_1, rho, and gamma to 1.0."""
        self.rho_1 = 1.0
        self.rho = 1.0
        self.gamma = 0.0
