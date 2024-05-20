import torch

from core.base import BaseDict
from utils.logging import setup_logger

logger = setup_logger(__name__)


class CriterionOuts(BaseDict):
    """Stores the output of the criterion.

    Attributes
    ----------
    loss : torch.Tensor
        Objective values.
    cw_loss : torch.Tensor
        CW loss values.
    softmax_cw_loss : torch.Tensor
        Loss values of the CW loss scaled by softmax function.
    grad : torch.Tensor
        Gradient of the objective function.
    target_class : torch.Tensor
        Class label with the 2nd highest classification probability.
    logit : torch.Tensor
        Logit (row output of the classification model)
    """

    def __init__(
        self,
        loss=None,
        cw_loss=None,
        softmax_cw_loss=None,
        grad=None,
        target_class=None,
        logit=None,
        *args,
        **kwargs,
    ) -> None:
        super(CriterionOuts, self).__init__(
            loss=loss,
            cw_loss=cw_loss,
            softmax_cw_loss=softmax_cw_loss,
            grad=grad,
            target_class=target_class,
            logit=logit,
            *args,
            **kwargs,
        )


@torch.no_grad()
def to_cpu(criterion_outs):
    """Create new instance on cpu

    Parameters
    ----------
    criterion_outs : CriterionOuts
        CriterionOuts instance to be send to cpu.

    Returns
    -------
    CriterionOuts
        New instance on cpu.
    """
    _grad = (
        criterion_outs.grad.detach().clone().cpu()
        if isinstance(criterion_outs.grad, torch.Tensor)
        else criterion_outs.grad
    )
    criterion_outs_cpu = CriterionOuts(
        loss=criterion_outs.loss.detach().clone().cpu(),
        cw_loss=criterion_outs.cw_loss.detach().clone().cpu(),
        softmax_cw_loss=criterion_outs.softmax_cw_loss.detach().clone().cpu(),
        grad=_grad,
        target_class=criterion_outs.target_class.detach().clone().cpu(),
        logit=criterion_outs.logit.detach().clone().cpu(),
    )
    return criterion_outs_cpu


class DiversityIndexContainer(BaseDict):
    """Retain latest K search points and their distance for diversity index calculation.

    Attributes
    ----------
    K: int
        Number of search points to retain.
    latest_K_points : torch.Tensor
        Latest K search points.
    DistanceMatrix : torch.Tensor
        Distance matrix of the retained search points.
    update_cnt : torch.Tensor
        Tensor to manage the indices that should be updated next.
    indices : torch.Tensor
        Tensor used to specify indices when updating the distance matrix.
    """

    def __init__(self, K: int, *args, **kwargs):
        """
        Parameters
        ----------
        K : int
            Number of search points to retain.
        """
        super(DiversityIndexContainer, self).__init__(
            K=K,
            latest_K_points=None,
            DistanceMatrix=None,
            update_cnt=None,
            indices=None,
            *args,
            **kwargs,
        )

    @torch.inference_mode()
    def push(self, xk: torch.Tensor):
        """Push new search points

        Parameters
        ----------
        xk : torch.Tensor
            Tensor of the k-th search points. size=[bs, channel, width, height]

        Returns
        -------
        bool
            True if there are K saved search points, False otherwise
        """
        bs = xk.shape[0]
        _new_point = xk.detach().view(bs, -1).unsqueeze(1).cpu().clone()

        if self.latest_K_points is None:
            self.update_cnt = torch.zeros((bs,), dtype=torch.long)
            self.indices = torch.arange(0, bs, dtype=torch.long)
            self.latest_K_points = _new_point  # bs, iteration, dimension
            self.DistanceMatrix = torch.zeros(size=(bs, 1, 1))

        elif self.latest_K_points.shape[1] < self.K:
            _shape = self.latest_K_points.shape
            incoming_change = (self.latest_K_points - _new_point.broadcast_to(size=_shape)).norm(
                p=2, dim=-1, keepdim=True
            )
            self.DistanceMatrix = torch.cat([self.DistanceMatrix, incoming_change], dim=-1)
            incoming_change = torch.cat([incoming_change, torch.zeros(size=(bs, 1, 1))], dim=1).permute(0, 2, 1)
            self.DistanceMatrix = torch.cat([self.DistanceMatrix, incoming_change], dim=1)
            self.latest_K_points = torch.cat([self.latest_K_points, _new_point], dim=1)

        else:
            _shape = self.latest_K_points.shape
            self.latest_K_points[self.indices, self.update_cnt, :] = _new_point.squeeze()
            incoming_change = (self.latest_K_points - _new_point.broadcast_to(size=_shape)).norm(p=2, dim=-1)
            self.DistanceMatrix[self.indices, self.update_cnt, :] = incoming_change
            self.DistanceMatrix[self.indices, :, self.update_cnt] = incoming_change

        self.update_cnt += 1
        self.update_cnt %= self.K

        return self.latest_K_points.shape[1] == self.K

    def clear(self):
        """Reset the attributes."""
        self.latest_K_points = None
        self.DistanceMatrix = None

        self.update_cnt = None
        self.indices = None


class Variable(BaseDict):
    """Stores and manages search points and their gradients.

    Attributes
    ----------
    xk : torch.Tensor
    """

    def __init__(
        self,
        xk=None,
        xk_1=None,
        sk=None,
        sk_1=None,
        gradk=None,
        gradk_1=None,
        *args,
        **kwargs,
    ):
        super(Variable, self).__init__(
            xk=xk,
            xk_1=xk_1,
            sk=sk,
            sk_1=sk_1,
            gradk=gradk,
            gradk_1=gradk_1,
            *args,
            **kwargs,
        )

    @torch.no_grad()
    def clone(self, device=torch.device("cpu")):
        """Create a copy

        Parameters
        ----------
        device : torch.device, optional
            Device location to put the copy, by default torch.device("cpu")

        Returns
        -------
        Variable
            Copy of this instance
        """
        clone_ = Variable()
        for key in self.keys():
            if isinstance(self[key], torch.Tensor):
                clone_[key] = self[key].detach().clone().to(device)
                assert id(clone_[key]) != id(self[key])
            else:
                clone_[key] = None
        return clone_

    def updateVariable(self, other, inds: torch.Tensor):
        """Update the attribute at the given index(inds)

        Parameters
        ----------
        other : Variable
            Another variable.
        inds : torch.Tensor
            Determine which elements to be updated.
        """
        if torch.logical_not(inds).all():
            return
        device = None
        is_same_device = False
        for key in self.keys():
            if other[key] is None:
                continue
            if device is None:
                device = self[key].device
                is_same_device = bool(device == other[key].device)
            tmp = other[key][inds].clone()
            if is_same_device:
                self[key][inds] = tmp.clone()
            else:
                self[key][inds] = tmp.clone().to(device)
            del tmp
            # assert (self[key][inds].cpu() == other[key][inds].cpu()).all()


class Information(BaseDict):
    """Gather and integrate the search information

    Attributes
    ----------
    best_loss : torch.Tensor
        Sequence of the best objective values at each iterations.
    best_cw_loss : torch.Tensor
        Sequence of the best CW loss values at each iterations.
    best_softmax_cw_loss : torch.Tensor
        Sequence of the best SCW loss values at each iterations.
    current_loss : torch.Tensor
        Sequence of the objective values at each iterations.
    current_cw_loss : torch.Tensor
        Sequence of the CW loss values at each iterations.
    current_softmax_cw_loss : torch.Tensor
        Sequence of the SCW loss values at each iterations.
    step_size : torch.Tensor
        Sequence of the step size at each iterations.
    diversity_index_1 : torch.Tensor
        Sequence of the Diversity Index (global) at each iterations >= K.
    diversity_index_2 : torch.Tensor
        Sequence of the Diversity Index (local) at each iterations >= K.
    dist_db : torch.Tensor
        Sequence of the estimated distance from dicision boundary at each iterations.
    target_class : torch.Tensor
        Sequence of the class labels with the 2nd highest classification probability at each iterations.
    n_projected_elms : torch.Tensor
        Sequence of the number of projected elements (out of the feasible region) at each iterations.
    n_boundary_elms : torch.Tensor
        Sequence of the number of elements on the boundary at each iterations.
    delta_x : torch.Tensor
        Sequence of euclidean distance between k-1 th and k-th search point at each iterations.
    grad_norm : torch.Tensor
        Sequence of euclidean norm of the search points at each iterations.
    x_adv : torch.Tensor
        The search point which has the highest objective value.
    grad_adv : torch.Tensor
        The gradient tensor at the x_adv.
    x_advs : torch.Tensor
        Sequence of each search points.
    """

    def __init__(
        self,
        step_size=None,
        diversity_index_1=None,
        diversity_index_2=None,
        target_class=None,
        n_projected_elms=None,
        n_boundary_elms=None,
        delta_x=None,
        grad_norm=None,
        best_loss=None,
        best_cw_loss=None,
        best_softmax_cw_loss=None,
        current_loss=None,
        current_cw_loss=None,
        current_softmax_cw_loss=None,
        x_adv=None,
        x_advs=None,
        grad_adv=None,
        *args,
        **kwargs,
    ):
        super(Information, self).__init__(
            step_size=step_size,
            diversity_index_1=diversity_index_1,
            diversity_index_2=diversity_index_2,
            target_class=target_class,
            n_projected_elms=n_projected_elms,
            n_boundary_elms=n_boundary_elms,
            delta_x=delta_x,
            grad_norm=grad_norm,
            best_loss=best_loss,
            best_cw_loss=best_cw_loss,
            best_softmax_cw_loss=best_softmax_cw_loss,
            current_loss=current_loss,
            current_cw_loss=current_cw_loss,
            current_softmax_cw_loss=current_softmax_cw_loss,
            x_adv=x_adv,
            x_advs=x_advs,
            grad_adv=grad_adv,
            *args,
            **kwargs,
        )

    def push_loss(
        self,
        indices: torch.Tensor,
        iteration: int,
        current_loss: torch.Tensor,
        current_cw_loss: torch.Tensor,
        current_softmax_cw_loss: torch.Tensor,
    ):
        """Update the loss values

        Parameters
        ----------
        iteration : int
            Current iteration of the search
        current_loss : torch.Tensor
            Current objective values
        current_cw_loss : torch.Tensor
            Current CW loss values
        current_softmax_cw_loss : torch.Tensor
            Current SCW loss values

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of indices corresponding to images with updated best objective function values.
        """
        self.current_loss[:, iteration + 1] = self.current_loss[:, iteration].clone()
        self.current_cw_loss[:, iteration + 1] = self.current_cw_loss[:, iteration].clone()
        self.current_softmax_cw_loss[:, iteration + 1] = self.current_softmax_cw_loss[:, iteration].clone()

        _current_loss = self.current_loss[indices].clone()
        _current_loss[:, iteration + 1] = current_loss.clone()
        self.current_loss[indices] = _current_loss.clone()
        _current_cw_loss = self.current_cw_loss[indices].clone()
        _current_cw_loss[:, iteration + 1] = current_cw_loss.clone()
        self.current_cw_loss[indices] = _current_cw_loss.clone()
        _current_softmax_cw_loss = self.current_softmax_cw_loss[indices].clone()
        _current_softmax_cw_loss[:, iteration + 1] = current_softmax_cw_loss.clone()
        self.current_softmax_cw_loss[indices] = _current_softmax_cw_loss.clone()

        if iteration == -1:  # initialization
            _best_loss = self.best_loss[indices].clone()
            _best_loss[:, iteration + 1] = current_loss.clone()
            self.best_loss[indices] = _best_loss.clone()
            _best_cw_loss = self.best_cw_loss[indices].clone()
            _best_cw_loss[:, iteration + 1] = current_cw_loss.clone()
            self.best_cw_loss[indices] = _best_cw_loss.clone()
            _best_softmax_cw_loss = self.best_softmax_cw_loss[indices].clone()
            _best_softmax_cw_loss[:, iteration + 1] = current_softmax_cw_loss.clone()
            self.best_softmax_cw_loss[indices] = _best_softmax_cw_loss.clone()
        else:
            self.best_loss[:, iteration + 1] = self.best_loss[:, iteration].clone()
            self.best_cw_loss[:, iteration + 1] = self.best_cw_loss[:, iteration].clone()
            self.best_softmax_cw_loss[:, iteration + 1] = self.best_softmax_cw_loss[:, iteration].clone()
            _best_loss = self.best_loss[indices].clone()
            best_loss = _best_loss[:, iteration].clone()
            ind = best_loss < current_loss
            best_loss[ind] = current_loss[ind].clone()
            _best_loss[:, iteration + 1] = best_loss.clone()
            self.best_loss[indices] = _best_loss.clone()

            _best_cw_loss = self.best_cw_loss[indices].clone()
            best_cw_loss = _best_cw_loss[:, iteration].clone()
            cw_ind = best_cw_loss < current_cw_loss
            best_cw_loss[cw_ind] = current_cw_loss[cw_ind].clone()
            _best_cw_loss[:, iteration + 1] = best_cw_loss.clone()
            self.best_cw_loss[indices] = _best_cw_loss.clone()

            _best_softmax_cw_loss = self.best_softmax_cw_loss[indices].clone()
            best_softmax_cw_loss = _best_softmax_cw_loss[:, iteration].clone()
            softmax_cw_ind = best_softmax_cw_loss < current_softmax_cw_loss
            best_softmax_cw_loss[softmax_cw_ind] = current_softmax_cw_loss[softmax_cw_ind].clone()
            _best_softmax_cw_loss[:, iteration + 1] = best_softmax_cw_loss.clone()
            self.best_softmax_cw_loss[indices] = _best_softmax_cw_loss.clone()

            return ind, cw_ind

    def push_best_x(self, v_best: Variable):
        """Add the adversarial examples

        Parameters
        ----------
        v_best : Variable
            Variables which have the highest objective values.
        """
        self["x_adv"] = v_best.xk.clone().cpu()
        self["grad_adv"] = v_best.gradk.clone().cpu()

    def push_advs(self, v: Variable):
        """Add search points at each iterations.

        Parameters
        ----------
        v : Variable
            Variables at the current iteration.
        """
        if self["x_advs"] is None:
            self["x_advs"] = v.xk.detach().clone().cpu()
        elif len(self.x_advs.shape) == len(v.xk.shape):
            self["x_advs"] = torch.stack([self["x_advs"], v.xk.detach().clone().cpu()], dim=0)
        else:
            self["x_advs"] = torch.cat([self["x_advs"], v.xk.detach().clone().cpu().unsqueeze(0)])
        logger.debug(f"{self.x_advs.shape}")

    def push_dict(self, indices: torch.Tensor, iteration: int, in_dict: dict):
        """Add search information contained in the input dictionary.

        Parameters
        ----------
        in_dict : dict
            A dictionary which contains several search information.

        Raises
        ------
        AttributeError
            Exception occurs when the unexpected information comes.
        """
        for key in in_dict:
            if in_dict[key] is None:
                continue
            elif self[key] is None:
                continue
            elif len(in_dict[key].shape) < 1:
                logger.warning(f"Tensor with invalid shape: {key}, {in_dict[key].shape}")
                continue
            assert isinstance(in_dict[key], torch.Tensor)
            if hasattr(self, key):
                # if self[key] is None:
                #     self[key] = in_dict[key].clone()
                # else:
                #     self[key] = torch.cat([self[key], in_dict[key].clone()], dim=1)
                # print(key, iteration, self[key].shape, indices.shape, in_dict[key].shape)
                logger.debug(f"{key}: {self[key][indices][:, iteration + 1].dtype}, {in_dict[key].dtype}")
                self[key][:, iteration + 1] = self[key][:, iteration].clone()
                tmp = self[key][indices].clone()
                tmp[:, iteration + 1] = in_dict[key].detach().clone()
                self[key][indices] = tmp.clone()
            else:
                raise AttributeError(f"{self.__class__.__name__} does not have {key}")

    def updateInformation(self, other, target_image_indices: torch.Tensor):
        """Update attributes at the given indices.

        Parameters
        ----------
        other : Information
            Information instance which has new values.
        target_image_indices : torch.Tensor
            Indices to be updated.
        """
        assert self.keys() == other.keys()
        for key in self.keys():
            if self[key] is None:
                logger.warning(f"self.{key} is None.")
                continue
            if other[key] is None:
                logger.warning(f"other.{key} is None.")
                continue
            else:
                if torch.isnan(other[key]).any():
                    logger.warning(f"nan in tensor: [ {key} ]")
                    other[key][torch.isnan(other[key])] = -1.0
                self[key][target_image_indices] = other[key].detach().clone()
            assert (self[key][target_image_indices] == other[key]).all()


class TargetLabelCollecter:
    def __init__(self, n_examples) -> None:
        self.y_targets = [set() for _ in range(n_examples)]

    def update(self, indices, targets):
        for ind in torch.where(indices)[0]:
            self.y_targets[ind] |= set(targets[ind].tolist())

    def to_list(self):
        return list(map(lambda x: list(x), self.y_targets))
