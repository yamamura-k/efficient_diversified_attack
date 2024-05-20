from typing import Tuple
import torch
import torch.nn as nn

from utils import setup_logger
from core.container import CriterionOuts

logger = setup_logger(__name__)


class Criterion:
    """Criterion of the adversarial attacks.

    Attributes
    ----------
    model : torch.nn.Module
        The attacked model
    criterions : Dict
        Dictionary of criterion instances.
    """

    def __init__(self, model):
        self.model = model
        self.criterions = dict(
            ce=CELoss(),
            cw=CWLoss(),
            targeted_ce=TargetedCELoss(),
            targeted_cw=TargetedCWLoss(),
            targeted_softmax_cw=TargetedSoftmaxCWLoss(),
            targeted_dlr=TargetedDLRLoss(),
            dlr=DLRLoss(),
            kld=KLDivLoss(),
            ods=ODS(),
            pas=PAS(),
            sigmoid_cw=SigmoidCWLoss(),
            softmax_cw=SoftmaxCWLoss(),
            max_cw=MaxCWLoss(),
            minmax_cw=MinmaxCWLoss(),
            g_dlr=GDLRLoss(),
            sum_cw=SumCWLoss(),
            sum_z=SumZLoss(),
        )

    def __call__(
        self,
        x,
        y,
        criterion_name="cw",
        enable_grad=True,
        scale=1.0,
        use_amp=True,
        *args,
        **kwargs,
    ):
        """Compute objective values and its gradient.

        Parameters
        ----------
        x : torch.Tensor
            Input of the attack target model
        y : torch.Tensor
            Ground truth class label
        criterion_name : str, optional
            The name of objective function, by default "cw"
        enable_grad : bool, optional
            Compute gradient if True, by default True
        _logit : torch.Tensor, optional
            Row output of the target model, by default None
        scale : float, optional
            Scaling factor of the logit, by default 1.0
        perturb : torch.Tensor, optional
            Difference between the clean input and current search point, by default None

        Returns
        -------
        CriterionOuts
            CriterionOuts instance.
        """
        with torch.set_grad_enabled(enable_grad):
            x.requires_grad_(enable_grad)
            with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=use_amp):
                logit = self.model(x) / scale
            x_sorted, ind_sorted = logit.sort(dim=1)
            ind = (ind_sorted[:, -1] == y).float()
            z_y = logit[torch.arange(logit.shape[0]), y]
            max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)

            loss = self.criterions[criterion_name](
                logit,
                y,
                z_y=z_y,
                max_zi=max_zi,
                x_sorted=x_sorted,
                ind_sorted=ind_sorted,
                *args,
                **kwargs,
            )

            grad = None
            if enable_grad:
                grad = torch.autograd.grad(loss.sum(), [x], retain_graph=False)[0]
                # grad = torch.autograd.grad(loss.sum(), [x], retain_graph=True)[0].detach().clone()
                # grad = torch.autograd.grad(loss.sum(), [x])[0].detach().clone()
        x.requires_grad_(False)

        with torch.inference_mode():
            target = ind * ind_sorted[:, -2] + (1 - ind) * ind_sorted[:, -1]
            cw_loss = self.criterions["cw"](logit, y, z_y=z_y, max_zi=max_zi, *args, **kwargs)
            softmax_cw_loss = self.criterions["softmax_cw"](logit, y, z_y=z_y, max_zi=max_zi, *args, **kwargs)
        if grad is not None:
            grad = grad.detach().clone()
        if loss is not None:
            loss = loss.detach().clone()
        criterion_outs = CriterionOuts(
            loss=loss,
            cw_loss=cw_loss.detach().clone(),
            softmax_cw_loss=softmax_cw_loss.detach().clone(),
            grad=grad,
            target_class=target.detach().clone().cpu(),
            # logit=logit.detach().clone().cpu(),
            acc=(y == logit.argmax(dim=1)).detach().clone(),
        )
        return criterion_outs


class CriterionManager(object):
    """Wrap Criterion class for more complicated loss construction.

    Attributes
    ----------
    criterion : Criterion
        Criterion instance.
    """

    def __init__(self, criterion: Criterion):
        self.criterion = criterion

    @torch.no_grad()
    def __call__(self, x, y, criterion_name, *args, **kwargs):
        """Compute output of the criterion

        Parameters
        ----------
        x : torch.Tensor
            Input of the attacked model.
        y : torch.Tensor
            Groud truth class label.
        criterion_name : str
            The name of criterion to be maximized.

        Returns
        -------
        CriterionOuts
        """
        _criterion_name = criterion_name.split("-")
        if len(_criterion_name) == 1:
            criterion_name, i, j = self.eval(criterion_name)
            return self.criterion(x, y, criterion_name, i=i, j=j, *args, **kwargs)
        else:
            assert len(_criterion_name) >= 3
            if _criterion_name[0] == "scale":
                assert len(_criterion_name) == 3
                scale = float(_criterion_name[1])
                return self.criterion(x, y, _criterion_name[-1], scale=scale, *args, **kwargs)
            else:
                raise NotImplementedError

    def eval(self, criterion_name: str):
        if "g_dlr" in criterion_name:
            name_1, name_2, i, j = criterion_name.split("_")
            i, j = int(i), int(j)
            criterion_name = "_".join([name_1, name_2])
        else:
            i, j = 0, 0
        return criterion_name, i, j


class Loss:
    def getName(self):
        """
        Returns
        -------
        name: str
            loss name
        """
        return self.name

    def getTarget(self):
        """return attack-target label, if it is untarget, then return None.

        Returns
        -------
        None or int (label)
        """
        return None


class UntargetedLoss(Loss):
    pass


class TargetedLoss(Loss):
    pass


class CELoss(UntargetedLoss):
    def __init__(self):
        self.name = "cross_entropy"
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def __call__(self, logits, y_true, *args, **kwargs):
        return self.cross_entropy(logits, y_true)


class TargetedCELoss(TargetedLoss):
    def __init__(self):
        self.name = "targeted_cross_entropy"
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def __call__(self, logits, y_true, y_target, *args, **kwargs):
        return self.cross_entropy(logits, y_target)


class DLRLoss(UntargetedLoss):
    """
    See Also
    --------
    https://arxiv.org/pdf/2003.01690.pdf
    """

    def __init__(self):
        self.name = "dlrloss"

    def __call__(self, logits, y_true, *args, **kwargs):
        return dlr_loss(x=logits, y=y_true, *args, **kwargs)


class TargetedDLRLoss(TargetedLoss):
    """
    See Also
    --------
    https://arxiv.org/pdf/2003.01690.pdf
    """

    def __init__(self):
        self.name = "dlrloss"

    def __call__(self, logits, y_true, y_target, *args, **kwargs):
        return targeted_dlr_loss(x=logits, y=y_true, y_adv=y_target, *args, **kwargs)


class ODS(UntargetedLoss):
    def __init__(self):
        self.name = "odsloss"

    def __call__(self, logits, y_true, w, *args, **kwargs):
        return ods(x=logits, w=w)


class PAS(UntargetedLoss):
    def __init__(self):
        self.name = "pasloss"

    def __call__(self, logits, y_true, w, *args, **kwargs):
        return just_idea(x=logits, y=y_true, w=w)


class CWLoss(UntargetedLoss):
    """
    See Also
    --------

    """

    def __init__(self):
        self.name = "cwloss"

    def __call__(self, logits, y_true, output_target_label=False, *args, **kwargs):
        return cw_loss(x=logits, y=y_true, output_target_label=output_target_label, *args, **kwargs)


class TargetedCWLoss(TargetedLoss):
    """
    See Also
    --------

    """

    def __init__(self):
        self.name = "targetedcwloss"

    def __call__(self, logits, y_true, y_target, *args, **kwargs):
        return targeted_cw_loss(x=logits, y=y_true, y_adv=y_target)


class KLDivLoss(Loss):
    def __init__(self):
        super().__init__()
        self.kld = nn.KLDivLoss(reduction="sum")

    def __call__(self, logits, y_true, adv_logits, *args, **kwargs):
        loss = self.kld(logits, adv_logits)
        return loss


# sigmoid_cw=SigmoidCWLoss(),
class SigmoidCWLoss(UntargetedLoss):
    def __init__(self):
        self.name = "sigmoidcwloss"

    def __call__(self, logits, y_true, output_target_label=False, *args, **kwargs):
        return sigmoid_cw_loss(x=logits, y=y_true, output_target_label=output_target_label, *args, **kwargs)


# softmax_cw=SoftmaxCWLoss(),
class SoftmaxCWLoss(UntargetedLoss):
    def __init__(self):
        self.name = "softmaxcwloss"

    def __call__(self, logits, y_true, output_target_label=False, *args, **kwargs):
        return softmax_cw_loss(x=logits, y=y_true, output_target_label=output_target_label, *args, **kwargs)


class TargetedSoftmaxCWLoss(TargetedLoss):
    def __init__(self):
        self.name = "targeted_softmaxcwloss"

    def __call__(self, logits, y_true, y_target, *args, **kwargs):
        return targeted_softmax_cw_loss(x=logits, y=y_true, y_adv=y_target)


# max_cw=MaxCWLoss(),
class MaxCWLoss(UntargetedLoss):
    def __init__(self):
        self.name = "maxcwloss"

    def __call__(self, logits, y_true, output_target_label=False, *args, **kwargs):
        return max_cw_loss(x=logits, y=y_true, output_target_label=output_target_label, *args, **kwargs)


# sum_cw=SumCWLoss(),
class SumCWLoss(UntargetedLoss):
    def __init__(self):
        self.name = "sumcwloss"

    def __call__(self, logits, y_true, output_target_label=False, *args, **kwargs):
        return sum_cw_loss(x=logits, y=y_true, output_target_label=output_target_label, *args, **kwargs)


class SumZLoss(UntargetedLoss):
    def __init__(self):
        self.name = "sumzloss"

    def __call__(self, logits, y_true, output_target_label=False, t=-3, *args, **kwargs):
        return sum_z_loss(
            x=logits,
            y=y_true,
            t=t,
            output_target_label=output_target_label,
            *args,
            **kwargs,
        )


# minmax_cw=MinmaxCWLoss(),
class MinmaxCWLoss(UntargetedLoss):
    def __init__(self):
        self.name = "minmaxcwloss"

    def __call__(self, logits, y_true, output_target_label=False, *args, **kwargs):
        return minmax_cw_loss(x=logits, y=y_true, output_target_label=output_target_label, *args, **kwargs)


# g_dlr=GDLRLoss()
class GDLRLoss(UntargetedLoss):
    def __init__(self):
        self.name = "scaledcwloss"

    def __call__(self, logits, y_true, output_target_label=False, i=1, j=2, *args, **kwargs):
        return g_dlr_loss(
            x=logits,
            y=y_true,
            output_target_label=output_target_label,
            i=i,
            j=j,
            *args,
            **kwargs,
        )


def dlr_loss(x, y, x_sorted=None, z_y=None, max_zi=None, *args, **kwargs):
    """
    Notes
    -----
    DLR = -\frac{z_y - \max_{i\neq y}z_i}{z_{\pi_1} - z_{\pi_3}}
    """
    if max_zi is None or x_sorted is None:
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)
    if z_y is None:
        z_y = x[torch.arange(x.shape[0]), y]

    # value_true_maximum = x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
    value_true_maximum = z_y - max_zi

    pi1_pi3 = x_sorted[:, -1] - x_sorted[:, -3] + 1e-6

    loss = (-1.0) * value_true_maximum / pi1_pi3
    return loss.reshape(-1)


def targeted_dlr_loss(x, y, y_adv, x_sorted=None, *args, **kwargs):
    """
    Notes
    -----
    DLR = -\frac{z_y - \max_{i\neq y}z_i}{z_{\pi_1} - z_{\pi_3}}
    """
    if x_sorted is None:
        x_sorted, ind_sorted = x.sort(dim=1)

    # ind = (ind_sorted[:, -1] == y).float()

    z_y = x[torch.arange(x.shape[0]), y]
    z_t = x[torch.arange(x.shape[0]), y_adv]

    value_true_maximum = z_y - z_t

    pi1_pi3_4 = x_sorted[:, -1] - (x_sorted[:, -3] + x_sorted[:, -4]) / 2 + 1e-6

    loss = (-1.0) * value_true_maximum / pi1_pi3_4
    return loss.reshape(-1)


def cw_loss(x, y, z_y=None, max_zi=None, output_target_label=False, *args, **kwargs):
    """
    Returns
    -------
    loss: torch.tensor
        shape is (torch.Size([n_batch])

    Notes
    -----
      -\frac{z_y - \max_{i\neq y}z_i}
    """
    if max_zi is None:
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)
    if z_y is None:
        z_y = x[torch.arange(x.shape[0]), y]

    # ind=1 (True)→ argmax_{i\neq y}(x) = -2
    # ind=0 (False)→ argmax_{i\neq y}(x) = -1

    # value_true_maximum = x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
    value_true_maximum = z_y - max_zi
    loss = (-1.0) * value_true_maximum

    if output_target_label and max_zi is None:
        target = ind * ind_sorted[:, -2] + (1 - ind) * ind_sorted[:, -1]
        return loss.reshape(-1), target.to(torch.int)
    else:
        return loss.reshape(-1)


def targeted_cw_loss(x, y, y_adv):
    """
    Returns
    -------
    loss: torch.tensor
        shape is (torch.Size([n_batch])

    Notes
    -----
      -\frac{z_y - \max_{i\neq y}z_i}
    """

    z_y = x[torch.arange(x.shape[0]), y]
    z_t = x[torch.arange(x.shape[0]), y_adv]
    value_true_maximum = z_y - z_t
    loss = (-1.0) * value_true_maximum
    return loss.reshape(-1)


def ods(x, w):
    return (x * w).sum(1)


def just_idea(x, y, w):
    z_y = x[torch.arange(x.shape[0]), y]  # .unsqueeze(1)
    z = x * w
    # loss = (z - z_y).clamp(min=0).sum(-1)
    # loss = (z - z_y).clamp(max=0).sum(-1)
    # loss = (z - z_y).sum(-1)
    # loss = -z.sum(1).pow(2) - z_y #.abs()
    # loss = -z.sum(1).pow(2) * torch.exp(-z_y) #.abs()
    loss = z.sum(1) * torch.exp(-z_y)
    # loss = z.sum(1) * torch.exp(-z_y) * torch.exp(x).sum(1)
    # loss = z.sum(1) * (-z_y + torch.log(torch.exp(x).sum(1)))
    return loss.reshape(-1)


def sigmoid_cw_loss(x, y, z_y=None, max_zi=None, output_target_label=False, *args, **kwargs):
    """
    Returns
    -------
    loss: torch.tensor
        shape is (torch.Size([n_batch])

    Notes
    -----
      -\frac{z_y - \max_{i\neq y}z_i}
    """
    if max_zi is None:
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)
    if z_y is None:
        z_y = x[torch.arange(x.shape[0]), y]

    # value_true_maximum = x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
    value_true_maximum = torch.sigmoid(z_y) - torch.sigmoid(max_zi)
    loss = (-1.0) * value_true_maximum

    if output_target_label and max_zi is None:
        target = ind * ind_sorted[:, -2] + (1 - ind) * ind_sorted[:, -1]
        return loss.reshape(-1), target.to(torch.int)
    else:
        return loss.reshape(-1)


def softmax_cw_loss(x, y, z_y=None, max_zi=None, output_target_label=False, *args, **kwargs):
    """
    Returns
    -------
    loss: torch.tensor
        shape is (torch.Size([n_batch])

    Notes
    -----
      -\frac{z_y - \max_{i\neq y}z_i}
    """
    if max_zi is None:
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)
    if z_y is None:
        z_y = x[torch.arange(x.shape[0]), y]

    # value_true_maximum = x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
    value_true_maximum = torch.exp(z_y) - torch.exp(max_zi)
    loss = (-1.0) * value_true_maximum / torch.exp(x).sum(1)

    if output_target_label and max_zi is None:
        target = ind * ind_sorted[:, -2] + (1 - ind) * ind_sorted[:, -1]
        return loss.reshape(-1), target.to(torch.int)
    else:
        return loss.reshape(-1)


def targeted_softmax_cw_loss(x, y, y_adv, output_target_label=False, *args, **kwargs):
    """
    Returns
    -------
    loss: torch.tensor
        shape is (torch.Size([n_batch])

    Notes
    -----
      -\frac{z_y - \max_{i\neq y}z_i}
    """
    z_y = x[torch.arange(x.shape[0]), y]
    z_t = x[torch.arange(x.shape[0]), y_adv]
    value_true_maximum = torch.exp(z_y) - torch.exp(z_t)
    loss = (-1.0) * value_true_maximum / torch.exp(x).sum(1)
    return loss.reshape(-1)


def max_cw_loss(x, y, z_y=None, max_zi=None, output_target_label=False, *args, **kwargs):
    """
    Returns
    -------
    loss: torch.tensor
        shape is (torch.Size([n_batch])

    Notes
    -----
      -\frac{z_y - \max_{i\neq y}z_i}
    """
    if max_zi is None:
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)
    if z_y is None:
        z_y = x[torch.arange(x.shape[0]), y]

    # value_true_maximum = x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
    value_true_maximum = z_y - max_zi
    loss = (-1.0) * value_true_maximum / (z_y + 1e-6)

    if output_target_label and max_zi is None:
        target = ind * ind_sorted[:, -2] + (1 - ind) * ind_sorted[:, -1]
        return loss.reshape(-1), target.to(torch.int)
    else:
        return loss.reshape(-1)


def sum_cw_loss(x, y, z_y=None, max_zi=None, output_target_label=False, *args, **kwargs):
    """
    Returns
    -------
    loss: torch.tensor
        shape is (torch.Size([n_batch])

    Notes
    -----
      -\frac{z_y - \max_{i\neq y}z_i}
    """
    if max_zi is None:
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)
    if z_y is None:
        z_y = x[torch.arange(x.shape[0]), y]

    # value_true_maximum = x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
    value_true_maximum = z_y - max_zi
    loss = (-1.0) * value_true_maximum / (z_y + max_zi)

    if output_target_label and max_zi is None:
        target = ind * ind_sorted[:, -2] + (1 - ind) * ind_sorted[:, -1]
        return loss.reshape(-1), target.to(torch.int)
    else:
        return loss.reshape(-1)


def sum_z_loss(x, y, t=3, x_sorted=None, *args, **kwargs):
    """

    Returns
    -------
    loss: torch.tensor
        shape is (torch.Size([n_batch])

    Notes
    -----
      -\frac{z_y - \max_{i\neq y}z_i}
    """
    if x_sorted is None:
        x_sorted, ind_sorted = x.sort(dim=1)

    ind = (ind_sorted[:, -1] == y).float()
    z_y = x[torch.arange(x.shape[0]), y]
    z_t = x_sorted[:, -t] * ind + x_sorted[:, -1] * (1.0 - ind)

    loss = (-1.0) * z_y / (z_y + z_t)
    return loss.reshape(-1)


def minmax_cw_loss(x, y, z_y=None, max_zi=None, output_target_label=False, *args, **kwargs):
    """
    Returns
    -------
    loss: torch.tensor
        shape is (torch.Size([n_batch])

    Notes
    -----
      -\frac{z_y - \max_{i\neq y}z_i}
    """
    if max_zi is None:
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)
    if z_y is None:
        z_y = x[torch.arange(x.shape[0]), y]

    # value_true_maximum = x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
    value_true_maximum = z_y - max_zi
    loss = (-1.0) * value_true_maximum / (x_sorted[:, -1] - x_sorted[:, 0] + 1e-6)

    if output_target_label and max_zi is None:
        target = ind * ind_sorted[:, -2] + (1 - ind) * ind_sorted[:, -1]
        return loss.reshape(-1), target.to(torch.int)
    else:
        return loss.reshape(-1)


def g_dlr_loss(
    x,
    y,
    z_y=None,
    max_zi=None,
    x_sorted=None,
    ind_sorted=None,
    output_target_label=False,
    i=1,
    j=0,
    *args,
    **kwargs,
):
    """
    Returns
    -------
    loss: torch.tensor
        shape is (torch.Size([n_batch])

    Notes
    -----
      -\frac{z_y - \max_{i\neq y}z_i}
    """
    if max_zi is None:
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        max_zi = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1.0 - ind)
    if z_y is None:
        z_y = x[torch.arange(x.shape[0]), y]

    # value_true_maximum = x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
    value_true_maximum = z_y - max_zi
    loss = (-1.0) * value_true_maximum / (x_sorted[:, -i] - x_sorted[:, -j] + 1e-6)

    if output_target_label and max_zi is None:
        target = ind * ind_sorted[:, -2] + (1 - ind) * ind_sorted[:, -1]
        return loss.reshape(-1), target.to(torch.int)
    else:
        return loss.reshape(-1)
