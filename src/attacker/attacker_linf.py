import torch

from utils import (
    setup_logger,
)
from core.projection import ProjectionLinf
from attacker.attacker import NormalAttacker


logger = setup_logger(__name__)


class AttackerLinf(NormalAttacker):
    def __init__(self, *args, **kwargs):
        super(AttackerLinf, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def setBounds(self, x_nat):
        self.upper = (x_nat + self.epsilon).clamp(0, 1).clone().to(self.device)
        self.lower = (x_nat - self.epsilon).clamp(0, 1).clone().to(self.device)
        assert isinstance(self.lower, torch.Tensor)

    @torch.no_grad()
    def setProjection(self, x_nat: torch.Tensor):
        self.setBounds(x_nat)
        self.projection = ProjectionLinf(lower=self.lower, upper=self.upper)

    def check_feasibility(self, x: torch.Tensor):
        assert (x >= self.lower.cpu()).all()
        assert (x <= self.upper.cpu()).all()
