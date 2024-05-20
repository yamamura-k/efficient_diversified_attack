import gc
import torch
from core.base import BaseAttacker
from core.algorithm import Algorithm
from core.stepsize import Stepsize
from core.condition import StepsizeCondition
from core.container import Information, Variable
from core.initial_point import InitialPoint
from metrics.diversity_index import DiversityIndex
from utils import updateParam, compute_information, setup_logger

logger = setup_logger(__name__)


class NormalAttacker(BaseAttacker):
    """White-box Attak

    Attributes
    ----------
    scale : float
        Scaling factor of the logit.
    device : torch.device
        cpu / cuda
    epsilon : float
        Radius of the fesible region.
    max_iter : int
        Maximum number of search iterations.
    export_level : int
        How many information to export
    algorithm : Algorithm
        Search point update formulations
    stepsize : Stepsize
        Stepsize update formulations
    stepsize_condition : StepsizeCondition
        Condition to update the stepsize
    diversity_index : DiversityIndex
        Compute diversity index
    initialpoint : InitialPoint
        Determine the initial search point x_0
    projection : Projection, None
        Projection function onto the feasible region
    lower : torch.Tensor
        Lower bound of the feasible region
    upper : torch.Tensor
        Upper bound of the feasible region
    """

    def __init__(
        self,
        max_iter,
        scale,
        epsilon,
        use_cw_value,
        num_nodes,
        param_algorithm,
        param_normalization,
        param_initialpoint,
        param_stepsize,
        device="cpu",
        export_level=20,
        *args,
        **kwargs,
    ) -> None:
        super(NormalAttacker, self).__init__()
        self.scale = scale
        self.device = device
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.export_level = export_level
        self.algorithm = Algorithm(**param_algorithm, **param_normalization)
        self.stepsize = Stepsize(
            device=device,
            **param_stepsize,
        )
        self.stepsize_condition = StepsizeCondition(
            device=device,
            use_cw_value=use_cw_value,
            **param_stepsize,
        )
        self.diversity_index = DiversityIndex(epsilon=epsilon, num_nodes=num_nodes)
        self.initialpoint = InitialPoint(**param_initialpoint, device=device, criterion=None)
        self.projection = None
        self.lower = None
        self.upper = None

    def updateParameters(
        self,
        max_iter,
        scale,
        epsilon,
        use_cw_value,
        num_nodes,
        param_algorithm,
        param_normalization,
        param_initialpoint,
        param_stepsize,
        device="cpu",
    ):
        """Update parameters of each attributes

        Parameters
        ----------
        max_iter : int
            Maximum number of search iterations
        scale : float
            Scaling factor of the logit
        epsilon : float
            Radius of the feasible region
        use_cw_value : bool
            Use CW loss values during the APGD's stepsize rule if True.
        num_nodes : int
            Number of search points to compute DI
        param_algorithm : dict
            Parameters of the algorithm instance
        param_normalization : dict
            Parameters of the normalization
        param_initialpoint : dict
            Parameters of the initialpoint instance
        param_stepsize : dict
            Parameters of the stepsize instance
        device : str, optional
            cpu / cuda, by default "cpu"
        """
        self.max_iter = max_iter
        self.scale = scale
        self.device = device
        updateParam(self.algorithm, param_algorithm)
        updateParam(self.algorithm.normalize, param_normalization)
        param_stepsize_cp = param_stepsize.copy()
        param_stepsize_cp.update(
            dict(
                max_iter=max_iter,
                device=device,
                epsilon=epsilon,
            )
        )
        updateParam(self.stepsize, param_stepsize_cp)
        param_condition = param_stepsize.copy()
        param_condition.update(dict(max_iter=max_iter, device=device, use_cw_value=use_cw_value))
        updateParam(self.stepsize_condition, param_condition)
        updateParam(self.initialpoint, param_initialpoint)
        updateParam(self.diversity_index, dict(epsilon=epsilon))
        updateParam(self.diversity_index.data_container, dict(K=num_nodes))

    @torch.no_grad()
    def initialize(
        self,
        x_best,
        x_nat,
        y_true,
        criterion,
        n_forward,
        n_backward,
    ):
        """Initialization of the search

        Parameters
        ----------
        x_best : torch.Tensor / None
            Best point
        x_nat : torch.Tensor
            Natural images
        y_true : torch.Tensor
            Ground truth label
        criterion : Criterion
            Objective function
        n_forward : int
            Number of the forward propagation
        n_backward : int
            Number of backward propagation

        Returns
        -------
        xk : torch.Tensor
            Initial point
        solution : Information
            Collect some metrics during the search
        n_forward : int
            Number of forward propagation
        n_backward : int
            Number of backward propagation
        begin : int
            The first iteration of the search
        """
        self.setProjection(x_nat)
        bs = x_nat.size(0)
        minus_ones = torch.full((bs, self.max_iter + 1), -1.0)
        minus_ones_int = torch.full((bs, self.max_iter + 1), -1, dtype=torch.int)

        solution = Information(
            best_loss=minus_ones.clone(),
            best_cw_loss=minus_ones.clone(),
            best_softmax_cw_loss=minus_ones.clone(),
            current_loss=minus_ones.clone(),
            current_cw_loss=minus_ones.clone(),
            current_softmax_cw_loss=minus_ones.clone(),
            step_size=minus_ones.clone() if self.export_level <= 40 else None,
            diversity_index_1=minus_ones.clone() if self.export_level <= 40 else None,
            diversity_index_2=minus_ones.clone() if self.export_level <= 40 else None,
            target_class=minus_ones_int.clone(),
            n_projected_elms=minus_ones_int.clone() if self.export_level <= 30 else None,
            n_boundary_elms=minus_ones_int.clone() if self.export_level <= 30 else None,
            delta_x=minus_ones.clone() if self.export_level <= 20 else None,
            grad_norm=minus_ones.clone() if self.export_level <= 20 else None,
        )
        updateParam(self.algorithm, dict(projection=self.projection))
        updateParam(self.initialpoint, dict(criterion=criterion))
        self.diversity_index.clear()
        self.stepsize_condition.reset()

        xk, fn, bn, begin = self.initialpoint(
            x_nat=x_nat,
            y_true=y_true,
            projection=self.projection,
            information=solution,
            diversity_index=self.diversity_index,
            x_best=x_best,
            export_level=self.export_level,
        )
        n_forward += fn
        n_backward += bn
        return xk, solution, n_forward, n_backward, begin

    @torch.no_grad()
    def attack(
        self,
        x_nat,
        y_true,
        criterion,
        criterion_name,
        algorithm_name,
        x_best=None,
        y_target=None,
        n_forward=0,
        n_backward=0,
        giveup=False,
        *args,
        **kwargs,
    ):
        """Generate the adversarial examples through maximize the objective

        Parameters
        ----------
        x_nat : torch.Tensor
            Natural image
        y_true : torch.Tensor
            Ground truth label
        criterion : Criterion
            Objective function
        criterion_name : str
            Name of objective function
        algorithm_name : str
            Name of algorithm to update the search points
        x_best : torch.Tensor, optional
            Incumbent solution of this problem, by default None
        y_target : torch.Tensor, optional
            Target class label for misclassification, by default None
        n_forward : int, optional
            Number of forward propagation, by default 0
        n_backward : int, optional
            Number of backward propagation, by default 0
        giveup : bool, optional
            Discard some images if True, by default False

        Returns
        -------
        solution : Information
            Search information
        n_forward : int
            Number of forward propagation
        n_backward : int
            Number of backward propagation
        accuracy : torch.Tensor
            Robust accuracy of the model after the attack on input images.
        """
        bs = x_nat.size(0)
        xk, solution, n_forward, n_backward, begin = self.initialize(
            x_best=x_best,
            x_nat=x_nat,
            y_true=y_true,
            criterion=criterion,
            n_forward=n_forward,
            n_backward=n_backward,
        )
        eta = torch.full((bs, 1, 1, 1), self.stepsize.initial_stepsize, device=self.device)
        attack_indices = torch.ones(bs, device=self.device, dtype=torch.bool)
        all_indices = torch.ones(bs, device=self.device, dtype=torch.bool)
        criterion_outs = criterion(
            xk,
            y_true,
            y_target=y_target,
            enable_grad=True,
            criterion_name=criterion_name,
            scale=self.scale,
        )
        gradk = criterion_outs.grad.clone()
        v = Variable(
            xk=xk.clone(),
            xk_1=xk.clone(),
            x_nes=xk.clone(),
            sk=gradk.clone(),
            sk_1=gradk.clone(),
            gradk=gradk.clone(),
            gradk_1=gradk.clone(),
            grad_nes=gradk.clone(),
        )
        v_best = v.clone(device=self.device)
        solution.push_loss(
            indices=attack_indices.cpu(),
            iteration=begin,
            current_loss=criterion_outs.loss.cpu(),
            current_cw_loss=criterion_outs.cw_loss.cpu(),
            current_softmax_cw_loss=criterion_outs.softmax_cw_loss.cpu(),
        )
        tmp_dict = compute_information(
            xk=v.xk,
            xk_1=v.xk_1,
            gradk=criterion_outs.grad,
            eta=eta.cpu(),
            lower=self.lower,
            upper=self.upper,
            target_class=criterion_outs.target_class,
            diversity_index=self.diversity_index,
            export_level=self.export_level,
            indices=attack_indices.cpu(),
        )
        solution.push_dict(indices=attack_indices.cpu(), iteration=begin, in_dict=tmp_dict)
        if self.export_level < 20:
            solution.push_advs(v)
        _algo_name = None
        algo_name = algorithm_name
        if algorithm_name == "NesCG":
            _algo_name = "GD"
            algo_name = "CG"
        elif "Nes" in algorithm_name:
            _algo_name = algorithm_name
            algo_name = "Nes2"
        is_nesterov = bool(algo_name == "Nes2")

        accuracy = criterion_outs.acc.detach()
        x_adv = xk.detach().clone()

        # eta = eta_0.clone() # if adaptive stepsize is applied, comment out
        for i in range(begin + 1, self.max_iter):
            # if giveup:
            #     attack_indices[attack_indices.clone()] = torch.logical_and(
            #         attack_indices[attack_indices.clone()],
            #         torch.logical_and(
            #             criterion_outs.acc, criterion_outs.cw_loss < 1e-3
            #         ),
            #     )

            if _algo_name is not None:
                n_forward, n_backward = self.PreUpdateNesterov(
                    y_true=y_true,
                    y_target=y_target,
                    v=v,
                    eta=eta,
                    criterion=criterion,
                    _algo_name=_algo_name,
                    criterion_name=criterion_name,
                    iteration=i,
                    attack_indices=attack_indices,
                    is_nesterov=is_nesterov,
                    n_forward=n_forward,
                    n_backward=n_backward,
                )

            _y_target = None if y_target is None else y_target[attack_indices]
            self.algorithm(v=v, eta=eta, algo_name=algo_name, iteration=i, indices=attack_indices)
            criterion_outs = criterion(
                v.xk[attack_indices],
                y_true[attack_indices],
                enable_grad=bool(1 - is_nesterov),
                criterion_name=criterion_name,
                scale=self.scale,
                y_target=_y_target,
            )
            n_forward += attack_indices.sum().item()
            n_backward += attack_indices.sum().item()
            if not is_nesterov:
                v.gradk[attack_indices] = criterion_outs.grad.clone()
            _update_inds = solution.push_loss(
                indices=attack_indices.cpu(),
                iteration=i,
                current_loss=criterion_outs.loss.cpu(),
                current_cw_loss=criterion_outs.cw_loss.cpu(),
                current_softmax_cw_loss=criterion_outs.softmax_cw_loss.cpu(),
            )
            update_inds = all_indices.clone()
            tmp_inds = update_inds[attack_indices]
            tmp_inds[_update_inds[self.stepsize_condition.use_cw_value]] = True
            update_inds[attack_indices] = tmp_inds.clone()

            v_best.updateVariable(v, update_inds)

            _inds = torch.logical_or(_update_inds[1].to(self.device), torch.logical_not(criterion_outs.acc))
            tmp = x_adv[attack_indices].detach().clone()
            tmp[_inds] = v.xk[attack_indices][_inds].clone()
            x_adv[attack_indices] = tmp.clone()
            del tmp

            update_stepsize_indices = self.stepsize_condition(l=solution, iteration=i, name=self.stepsize.strategy, v=v)
            eta = self.stepsize(
                iteration=i,
                eta=eta,
                inds=update_stepsize_indices,
                v=v,
                v_best=v_best,
            )
            if self.stepsize.strategy == "apgd" and isinstance(update_stepsize_indices, torch.Tensor):
                assert (v.xk[update_stepsize_indices] == v_best.xk[update_stepsize_indices]).all()
            accuracy[attack_indices] = torch.logical_and(accuracy[attack_indices], criterion_outs.acc)
            logger.info(f"[ #attacked images ] {attack_indices.sum().item()}")
            logger.info(
                f"[ {algo_name:^4} ] iteration {i:>4}: {criterion_outs.cw_loss.sum().item():.4f}, {solution.best_cw_loss[:, i + 1].sum().item():.4f}"
            )
            tmp_dict = compute_information(
                xk=v.xk,
                xk_1=v.xk_1,
                gradk=v.gradk[attack_indices],
                eta=eta[attack_indices].cpu(),
                lower=self.lower,
                upper=self.upper,
                target_class=criterion_outs.target_class,
                diversity_index=self.diversity_index,
                export_level=self.export_level,
                indices=attack_indices.cpu(),
            )
            solution.push_dict(indices=attack_indices.cpu(), iteration=i, in_dict=tmp_dict)
            if (eta == 0).all() or torch.logical_not(attack_indices).all():
                logger.warning(f"{algo_name} is converged at iteration {i}")
                for j in range(i + 1, self.max_iter):
                    solution.push_loss(
                        indices=attack_indices.cpu(),
                        iteration=j,
                        current_loss=criterion_outs.loss.clone().cpu(),
                        current_cw_loss=criterion_outs.cw_loss.clone().cpu(),
                        current_softmax_cw_loss=criterion_outs.softmax_cw_loss.clone().cpu(),
                    )
                    solution.push_dict(indices=attack_indices.cpu(), iteration=j, in_dict=tmp_dict)
                break
            if (v.xk == v.xk_1).all() and i > 0:
                logger.warning(f"xk == xk_1 at iteration {i}")
            if (v_best.xk == v_best.xk_1).all() and i > 0:
                logger.warning(f"xk_best == xk_1_best at iteration {i}")
            if (v.gradk == v.gradk_1).all() and i > 0:
                logger.warning(f"gradk == gradk_1 at iteration {i}")
            if self.export_level < 20:
                solution.push_advs(v)
            gc.collect()

        solution.x_adv = x_adv.detach().clone().cpu()
        logger.info(f"#forward: {n_forward}, #backward: {n_backward}")
        self.check_feasibility(solution.x_adv)
        del v, v_best
        gc.collect()
        return solution, n_forward, n_backward, accuracy.cpu()

    @torch.no_grad()
    def PreUpdateNesterov(
        self,
        y_true,
        y_target,
        v,
        eta,
        criterion,
        _algo_name,
        criterion_name,
        iteration,
        attack_indices,
        is_nesterov,
        n_forward,
        n_backward,
    ):
        """Compute x_nes and its gradient

        Parameters
        ----------
        x_nat : torch.Tensor
            Natural image
        y_true : torch.Tensor
            Ground truth label
        y_target : torch.Tensor
            Target class label for misclassification
        v : Variable
            Contains the current & previous search points and its gradients
        criterion_outs : str
            The latest output of the objective function
        solution : Information
            Collect information during the search
        eta : torch.Tensor
            Current step size
        criterion : Criterion
            Objective function
        _algo_name : str
            The way to compute x_nes
        criterion_name : str
            Name of the objective function
        iteration : int
            Current number of search iterations
        attack_indices : torch.Tensor
            Determine the attack image
        inds : torch.Tensor
            Used in advanced criterion
        is_nesterov : bool
            If Thue, the main algorithm is Nesterov's Acceralated Gradient
        n_forward : int
            Number of forward propagation
        n_backward : int
            Number of backward propagation

        Returns
        -------
        n_forward : int
            Number of forward propagation
        n_backward : int
            Number of backward propagation
        """
        _y_target = None if y_target is None else y_target[attack_indices]
        x_tmp = self.algorithm(
            v=v,
            eta=eta,
            algo_name=_algo_name,
            iteration=iteration,
            indices=attack_indices,
            update_v=False,
        )
        criterion_outs_nes = criterion(
            x_tmp[attack_indices],
            y_true[attack_indices],
            enable_grad=True,
            criterion_name=criterion_name,
            scale=self.scale,
            y_target=_y_target,
        )
        n_forward += attack_indices.sum().item()
        n_backward += attack_indices.sum().item()
        if is_nesterov:
            v.grad_nes[attack_indices] = criterion_outs_nes.grad.clone()
        v.gradk[attack_indices] = criterion_outs_nes.grad.clone()
        return n_forward, n_backward

    @torch.no_grad()
    def setProjection(self, x: torch.Tensor):
        self.projection = lambda x: x.clamp(min=0.0, max=1.0)

    @torch.no_grad()
    def check_feasibility(self, x: torch.Tensor):
        assert (x >= 0.0).all()
        assert (x <= 1.0).all()
