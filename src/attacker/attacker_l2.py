import torch

from utils import (
    setup_logger,
)
from core.projection import ProjectionL2
from attacker.attacker import NormalAttacker


logger = setup_logger(__name__)


class AttackerL2(NormalAttacker):
    def __init__(self, *args, **kwargs):
        super(AttackerL2, self).__init__(*args, **kwargs)

    def setProjection(self, x_nat: torch.Tensor):
        self.x_nat = x_nat.clone()
        self.projection = ProjectionL2(epsilon=self.epsilon, x_nat=x_nat, _min=0.0, _max=1.0)

    def check_feasibility(self, x: torch.Tensor):
        assert ((x - self.x_nat.cpu()).norm(p=2, dim=(1, 2, 3)) <= self.epsilon + 1e-5).all()
        assert (x >= 0.0).all()
        assert (x <= 1.0).all()
