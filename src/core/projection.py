import torch


class Projection(object):
    def __init__(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def __call__(self, xk: torch.Tensor):
        return xk


class ProjectionLinf(Projection):
    def __init__(self, lower: torch.Tensor = None, upper: torch.Tensor = None):
        self.upper = upper
        self.lower = lower

    @torch.no_grad()
    def __call__(self, xk: torch.Tensor):
        xk = torch.minimum(torch.maximum(xk, self.lower), self.upper)
        assert (xk >= self.lower).all().item(), f"{(xk < self.lower).sum().item()}"
        assert (xk <= self.upper).all().item(), f"{(xk > self.upper).sum().item()}"
        return xk

    def setBounds(self, lower: torch.Tensor, upper: torch.Tensor):
        self.upper = upper.clone()
        self.lower = lower.clone()


class ProjectionL2(Projection):
    def __init__(self, epsilon, x_nat, _min=0, _max=1):
        super().__init__()
        self.epsilon = epsilon
        self.x_nat = x_nat.clone()
        self._min = _min
        self._max = _max
        self.upper = None
        self.lower = None

    @torch.no_grad()
    def __call__(self, xk: torch.Tensor):
        perturb = xk - self.x_nat
        norm = perturb.norm(p=2, dim=(1, 2, 3), keepdim=True)
        perturb = perturb * torch.minimum(torch.tensor(1.0), self.epsilon / (norm + 1e-12))
        xk = self.x_nat + perturb
        xk = xk.clamp(min=self._min, max=self._max)
        assert (
            perturb.norm(p=2, dim=(1, 2, 3)) <= self.epsilon + 1e-5
        ).all(), f"{perturb.norm(p=2, dim=(1, 2, 3)).min()}, {perturb.norm(p=2, dim=(1, 2, 3)).mean()}, {perturb.norm(p=2, dim=(1, 2, 3)).max()},{(perturb.norm(p=2, dim=(1, 2, 3)) - self.epsilon).max()}, {((perturb.norm(p=2, dim=(1, 2, 3)) > self.epsilon).sum())}"
        _perturb = xk - self.x_nat
        assert (
            _perturb.norm(p=2, dim=(1, 2, 3)) <= self.epsilon + 1e-5
        ).all(), f"{_perturb.norm(p=2, dim=(1, 2, 3)).min()}, {_perturb.norm(p=2, dim=(1, 2, 3)).mean()}, {_perturb.norm(p=2, dim=(1, 2, 3)).max()},{(_perturb.norm(p=2, dim=(1, 2, 3)) - self.epsilon).max()}, {((_perturb.norm(p=2, dim=(1, 2, 3)) > self.epsilon).sum())}"
        assert (xk >= self._min).all()
        assert (xk <= self._max).all()
        return xk


# class ProjectionL1(Projection):
#     def __init__(self, epsilon, x_nat, _min=0, _max=1):
#         super().__init__()
#         self.epsilon = epsilon
#         self.x_nat = x_nat.clone()
#         self._min = _min
#         self._max = _max
#         self.upper = None
#         self.lower = None

#     @torch.no_grad()
#     def __call__(self, xk: torch.Tensor):
#         perturb = xk - self.x_nat
#         norm = perturb.norm(p=1, dim=(1, 2, 3), keepdim=True)
#         perturb = perturb * torch.minimum(torch.tensor(1.0), self.epsilon / (norm + 1e-12))
#         xk = self.x_nat + perturb
#         xk = xk.clamp(min=self._min, max=self._max)
#         assert (perturb.norm(p=1, dim=(1, 2, 3)) <= self.epsilon + 1e-3).all(), f"{perturb.norm(p=1, dim=(1, 2, 3)).min()}, {perturb.norm(p=1, dim=(1, 2, 3)).mean()}, {perturb.norm(p=1, dim=(1, 2, 3)).max()},{(perturb.norm(p=1, dim=(1, 2, 3)) - self.epsilon).max()}, {((perturb.norm(p=1, dim=(1, 2, 3)) > self.epsilon).sum())}"
#         _perturb = xk - self.x_nat
#         assert (_perturb.norm(p=1, dim=(1, 2, 3)) <= self.epsilon + 1e-3).all(), f"{_perturb.norm(p=1, dim=(1, 2, 3)).min()}, {_perturb.norm(p=1, dim=(1, 2, 3)).mean()}, {_perturb.norm(p=1, dim=(1, 2, 3)).max()},{(_perturb.norm(p=1, dim=(1, 2, 3)) - self.epsilon).max()}, {((_perturb.norm(p=1, dim=(1, 2, 3)) > self.epsilon).sum())}"
#         assert (xk >= self._min).all()
#         assert (xk <= self._max).all()
#         return xk
