import torch
from utils import setup_logger, compute_information

logger = setup_logger(__name__)


class InitialPoint:
    def __init__(
        self,
        method: str,
        dataset: str,
        epsilon: float,
        odi_step: float,
        odi_iter: int,
        device: torch.device,
        criterion,
    ):
        self.method = method
        self.epsilon = epsilon
        self.dataset = dataset
        self.odi_step = odi_step
        self.odi_iter = odi_iter
        self.device = device
        self.criterion = criterion

    @torch.no_grad()
    def __call__(
        self,
        x_nat,
        y_true,
        projection,
        criterion_name="ce",
        x_best=None,
        *args,
        **kwargs,
    ):
        n_forward = n_backward = 0
        begin = -1
        if self.method == "center":
            return (
                ((projection.lower + projection.upper) / 2).to(self.device),
                n_forward,
                n_backward,
                begin,
            )
        elif self.method == "random":
            return self.getRandomInput(projection), n_forward, n_backward, begin
        elif self.method == "odi":
            n_forward = n_backward = self.odi_iter * x_nat.size(0)
            begin += self.odi_iter
            return (
                self.OutputDiversifiedSampling(x_nat, y_true, projection, *args, **kwargs),
                n_forward,
                n_backward,
                begin,
            )
        elif self.method == "pas":
            n_forward = n_backward = self.odi_iter * x_nat.size(0)
            begin += self.odi_iter
            return (
                self.PredictionAwareSampling(x_nat, y_true, projection, *args, **kwargs),
                n_forward,
                n_backward,
                begin,
            )
        elif self.method == "random-odi":
            x_rand, n_forward, n_backward, begin = self.getRandomInput(projection), n_forward, n_backward, begin
            n_forward += self.odi_iter * x_nat.size(0)
            n_backward += self.odi_iter * x_nat.size(0)
            begin += self.odi_iter
            return (
                self.OutputDiversifiedSampling(x_rand, y_true, projection, *args, **kwargs),
                n_forward,
                n_backward,
                begin,
            )
        elif self.method == "random-pas":
            x_rand, n_forward, n_backward, begin = self.getRandomInput(projection), n_forward, n_backward, begin
            n_forward += self.odi_iter * x_nat.size(0)
            n_backward += self.odi_iter * x_nat.size(0)
            begin += self.odi_iter
            return (
                self.PredictionAwareSampling(x_rand, y_true, projection, *args, **kwargs),
                n_forward,
                n_backward,
                begin,
            )
        elif self.method == "input":
            return x_nat.clone().to(self.device), n_forward, n_backward, begin
        elif self.method == "center-input":
            n_forward = x_nat.size(0) * 2
            center = self("center", x_nat, y_true, projection)
            criterion_outs_center = self.criterion(center, y_true, "cw", enable_grad=False)
            criterion_outs_input = self.criterion(x_nat.clone().to(self.device), y_true, "cw", enable_grad=False)
            cw_center = criterion_outs_center.cw_loss
            cw_input = criterion_outs_input.cw_loss
            center[cw_input > cw_center] = x_nat[cw_input > cw_center].clone().to(self.device)
            return center.clone().to(self.device), n_forward, n_backward, begin
        elif self.method == "opt-based":
            n_forward = n_backward = self.odi_iter * x_nat.size(0)
            begin += self.odi_iter
            return (
                self.OptBased(
                    x_nat,
                    y_true,
                    projection,
                    criterion_name=criterion_name,
                    **args,
                    **kwargs,
                ),
                n_forward,
                n_backward,
                begin,
            )
        elif self.method in {"best", "storage"} and x_best is not None:
            return x_best, n_forward, n_backward, begin
        else:
            raise NotImplementedError(f"[ Initial Point ] {self.method} is not implemented")

    def getRandomInput(self, projection):
        width = projection.upper - projection.lower
        perturb = torch.rand_like(width, device=self.device)
        x_init = width * perturb + projection.lower
        return x_init

    def OutputDiversifiedSampling(self, x_nat, y_true, projection, information, diversity_index, export_level=20):
        bs = x_nat.shape[0]
        indices = torch.ones((bs,), dtype=torch.bool)
        # indices = torch.ones((bs,), device=self.device, dtype=torch.bool)
        # FIXME: fix seed
        if self.dataset == "imagenet":
            output_dim = 1000
        elif self.dataset in {"cifar10", "mnist"}:
            output_dim = 10
        elif self.dataset == "cifar100":
            output_dim = 100
        else:
            raise NotImplementedError(f"ODS for dataset {self.dataset} is not implemented.")
        w = torch.empty((bs, output_dim), device=self.device).uniform_(-1, 1)

        x = x_nat.clone().to(self.device)
        x_1 = x.clone()
        for i in range(self.odi_iter):
            criterion_outs = self.criterion(x, y_true, w=w, criterion_name="ods", enable_grad=True)
            vods = criterion_outs.grad / (criterion_outs.grad.norm(p=2, dim=(1, 2, 3), keepdim=True).pow(2) + 1e-12)

            information.push_loss(
                indices=indices,
                iteration=i - 1,
                current_loss=criterion_outs.loss.cpu(),
                current_cw_loss=criterion_outs.cw_loss.cpu(),
                current_softmax_cw_loss=criterion_outs.softmax_cw_loss.cpu(),
            )
            tmp_dict = compute_information(
                xk=x,
                xk_1=x_1,
                gradk=criterion_outs.grad,
                eta=torch.full((bs,), self.epsilon).cpu(),
                lower=projection.lower,
                upper=projection.upper,
                target_class=criterion_outs.target_class,
                diversity_index=diversity_index,
                export_level=export_level,
                indices=indices,
            )
            information.push_dict(indices=indices, iteration=i - 1, in_dict=tmp_dict)

            x = projection(x + self.odi_step * torch.sign(vods))
            x_1 = x.clone()
            logger.debug(f"[ ODS iteration {i} ]: cw loss = {criterion_outs.cw_loss.mean().item():.4f}")
        return x.detach().clone()

    def PredictionAwareSampling(self, x_nat, y_true, projection, information, diversity_index, export_level=20):
        bs = x_nat.shape[0]
        indices = torch.ones((bs,), dtype=torch.bool)
        # indices = torch.ones((bs,), device=self.device, dtype=torch.bool)
        # FIXME: fix seed
        if self.dataset == "imagenet":
            output_dim = 1000
        elif self.dataset in {"cifar10", "mnist"}:
            output_dim = 10
        elif self.dataset == "cifar100":
            output_dim = 100
        else:
            raise NotImplementedError(f"ODS for dataset {self.dataset} is not implemented.")
        w = torch.empty((bs, output_dim), device=self.device).uniform_(-1, 1)

        x = x_nat.clone().to(self.device)
        x_1 = x.clone()
        for i in range(self.odi_iter):
            criterion_outs = self.criterion(x, y_true, w=w, criterion_name="pas", enable_grad=True)
            vods = criterion_outs.grad / (criterion_outs.grad.norm(p=2, dim=(1, 2, 3), keepdim=True).pow(2) + 1e-12)

            information.push_loss(
                indices=indices,
                iteration=i - 1,
                current_loss=criterion_outs.loss.cpu(),
                current_cw_loss=criterion_outs.cw_loss.cpu(),
                current_softmax_cw_loss=criterion_outs.softmax_cw_loss.cpu(),
            )
            tmp_dict = compute_information(
                xk=x,
                xk_1=x_1,
                gradk=criterion_outs.grad,
                eta=torch.full((bs,), self.epsilon).cpu(),
                lower=projection.lower,
                upper=projection.upper,
                target_class=criterion_outs.target_class,
                diversity_index=diversity_index,
                export_level=export_level,
                indices=indices,
            )
            information.push_dict(indices=indices, iteration=i - 1, in_dict=tmp_dict)

            x = projection(x + self.odi_step * torch.sign(vods))
            x_1 = x.clone()
            logger.debug(f"[ PredictionAwareSampling iteration {i} ]: cw loss = {criterion_outs.cw_loss.mean().item():.4f}")
        return x.detach().clone()

    def OptBased(
        self,
        x_nat,
        y_true,
        projection,
        information,
        diversity_index,
        criterion_name="ce",
        export_level=20,
    ):
        bs = x_nat.size(0)
        # indices = torch.ones((bs,), device=self.device, dtype=torch.bool)
        indices = torch.ones((bs,), dtype=torch.bool)
        x = x_nat.clone().to(self.device)
        x0 = x_nat.clone().to(self.device)
        x_1 = x.clone()
        for i in range(self.odi_iter):
            criterion_outs = self.criterion(
                x,
                y_true,
                criterion_name=criterion_name,
                enable_grad=True,
                perturb=(x - x0).clone(),
            )

            information.push_loss(
                indices=indices,
                iteration=i - 1,
                current_loss=criterion_outs.loss.cpu(),
                current_cw_loss=criterion_outs.cw_loss.cpu(),
                current_softmax_cw_loss=criterion_outs.softmax_cw_loss.cpu(),
            )
            tmp_dict = compute_information(
                xk=x,
                xk_1=x_1,
                gradk=criterion_outs.grad,
                eta=torch.full((bs,), self.epsilon).cpu(),
                lower=projection.lower,
                upper=projection.upper,
                target_class=criterion_outs.target_class,
                diversity_index=diversity_index,
                export_level=export_level,
                indices=indices,
            )
            information.push_dict(indices=indices, iteration=i - 1, in_dict=tmp_dict)

            x = projection(x + self.odi_step * torch.sign(criterion_outs.grad))
            x_1 = x.clone()
            logger.debug(
                f"[ OptBase ({criterion_name}) iteration {i} ]: cw loss = {criterion_outs.cw_loss.mean().item():.4f}"
            )
        return x.detach().clone()
