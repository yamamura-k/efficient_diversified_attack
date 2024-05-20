import os
import yaml
import torch
from core.base import BaseDict
from .logging import setup_logger

logger = setup_logger(__name__)


def read_yaml(paths):
    """reader of yaml file
    Parameters
    ----------
    paths: str or list of str
        yaml file path(s)

    Returns
    -------
    obj: BaseDict
    """
    if isinstance(paths, str):
        paths = [paths]

    obj = BaseDict()
    for path in paths:
        logger.debug(f"\n [ READ ] {path}")
        f = open(path, mode="r")
        _obj = yaml.safe_load(f)
        f.close()

        for key, value in _obj.items():
            if not isinstance(value, dict):
                continue
            if not hasattr(value, "__dict__"):
                tmp = BaseDict()
                tmp.update(value.copy())
                _obj[key] = tmp.copy()
            logger.info(f"\n {key} ← {value}")
        obj.update(_obj.copy())
    return obj


def overwrite_config(cmd_param, config):
    """update parameter dict
    Parameters
    ----------
    cmd_param : list of str
        each element is param:cast:value
    config : dict
        parameter dict
    """
    for param_set in cmd_param:
        param, cast, value = param_set.split(":")
        param_layer = param.split(".")
        param_dict = config
        update = True
        for param_name in param_layer[:-1]:
            if param_name not in param_dict:
                param_dict[param_name] = BaseDict()
                update = False
            param_dict = param_dict[param_name]
        if param_layer[-1] not in param_dict:
            update = False
        if cast == "bool":
            assert value in {
                "True",
                "False",
            }, "the param type bool must be True or False."
            param_dict[param_layer[-1]] = False if value == "False" else True
        elif "list" in cast:
            _cast = eval(cast.split("_")[1])
            value = list(map(lambda x: _cast(x), value.split(",")))
            param_dict[param_layer[-1]] = value
        else:
            param_dict[param_layer[-1]] = eval(f'{cast}("{value}")')
        logger.info(f'{["new", "update"][update]} param {param_layer[-1]} <- {value}')
    return config


class InformationWriter(object):
    """Export Information instance to csv / pth files.

    Attributes
    ----------
    export_level : int
        Determine which data to be exported.
    export_data : Dict[Set]
        Determine which data to be exported.
    """

    def __init__(self, export_level: int = 70):
        super(InformationWriter, self).__init__()
        assert export_level % 10 == 0, f"export_level {export_level} is invalid: export_level%10 should be 0."
        self.export_level = export_level
        self.export_data = {
            60: set(["best_loss", "best_cw_loss", "best_softmax_cw_loss"]),
            50: set(
                [
                    "current_loss",
                    "current_cw_loss",
                    "current_softmax_cw_loss",
                ]
            ),
            40: set(
                [
                    "step_size",
                    "target_class",
                    "diversity_index_1",
                    "diversity_index_2",
                ]
            ),
            30: set(
                [
                    "n_projected_elms",
                    "n_boundary_elms",
                ]
            ),
            20: set(
                [
                    "delta_x",
                    "grad_norm",
                ]
            ),
            10: set(
                [
                    "x_adv",
                    "x_advs",
                    # "grad_adv",
                ]
            ),
        }

    def setLevel(self, export_level: int):
        self.export_level = export_level

    def __call__(self, information, save_dir: str):
        """Export Information

        Parameters
        ----------
        information : Information
            Information instance to be exported.
        save_dir : str
            Path to the exported files.
        """
        if self.export_level > 60:
            return
        else:
            for level in range(self.export_level, 70, 10):
                for key in self.export_data[level]:
                    if information[key] is None:
                        logger.warning(f"information.{key} is None.")
                        continue
                    else:
                        if len(information[key].shape) <= 2:
                            save_path = os.path.join(save_dir, f"{key}.csv")
                            tensor2csv(information[key], save_path)
                        else:
                            save_path = os.path.join(save_dir, f"{key}.pth")
                            torch.save(information[key], save_path)
                        logger.debug(f"[ SAVE ] {save_path}")


def tensor2csv(tensor: torch.Tensor, save_path: str):
    f = open(save_path, mode="a")
    bs = tensor.shape[0]
    dim = len(tensor.shape)
    if dim == 1:
        tensor = tensor.unsqueeze(1)
    elif dim > 2:
        raise ValueError(f"Tensor dim {dim} must be smaller than or equal to 2.")
    for j in range(bs):
        j_th_tensor = tensor[j, :]
        j_th_list = j_th_tensor.tolist()
        if j_th_tensor.dtype in {torch.float16, torch.float32, torch.float64}:
            string = ",".join(map(lambda x: "{:.5f}".format(x), j_th_list))
        else:
            string = ",".join(map(str, j_th_list))
        print(string, file=f)

    f.close()


@torch.no_grad()
def compute_information(
    xk: torch.Tensor,
    xk_1: torch.Tensor,
    gradk: torch.Tensor,
    eta: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    target_class: torch.Tensor,
    diversity_index,
    indices,
    export_level: int = 60,
):
    """Compute search information to be stored.

    Parameters
    ----------
    xk : torch.Tensor
        k-th search point
    xk_1 : torch.Tensor
        k-1 th serch point
    gradk : torch.Tensor
        k-th gradient
    eta : torch.Tensor
        Step size
    lower : torch.Tensor
        Lower bound of the feasible region
    upper : torch.Tensor
        Upper bound of the feasible region
    target_class : torch.Tensor
        Class label with 2nd highest classification probability
    diversity_index : DiversityIndex
        Compute Diversity Index
    dist_db : torch.Tensor
        Estimated distance from xk to the decision boundary
    indices : torch.Tensor
        Bool tensor for indexing
    export_level : int, optional
        export level, by default 60

    Returns
    -------
    Dict
        Dictionary of the calculated search information.
    """
    if export_level <= 30 and isinstance(lower, torch.Tensor) and isinstance(upper, torch.Tensor):
        lower = lower.cpu()
        upper = upper.cpu()
        xk = xk.cpu()
        xk_1 = xk_1.cpu()
        n_projected_elms = ((xk - xk_1).abs() < 1e-12).sum(dim=(1, 2, 3)).to(torch.int)[indices]
        n_boundary_elms = (
            torch.logical_or((xk - lower).abs() < 1e-12, (upper - xk).abs() < 1e-12)
            .sum(dim=(1, 2, 3))
            .to(torch.int)[indices]
        )
    else:
        n_projected_elms = None
        n_boundary_elms = None
    if export_level <= 30:
        diversity_index_1, diversity_index_2 = diversity_index(xk, eta)
        if diversity_index_1 is not None:
            diversity_index_1 = diversity_index_1.clone()[indices]
        if diversity_index_2 is not None:
            diversity_index_2 = diversity_index_2.clone()[indices]
    else:
        diversity_index_1, diversity_index_2 = None, None
    if export_level <= 20:
        gradk = gradk.cpu()
        delta_x = (xk - xk_1).norm(p=2, dim=(1, 2, 3))[indices]
        grad_norm = gradk.norm(p=2, dim=(1, 2, 3))
    else:
        delta_x = None
        grad_norm = None

    ret_dict = dict(
        step_size=eta.squeeze(),
        target_class=target_class.to(torch.int),
        n_projected_elms=n_projected_elms,
        n_boundary_elms=n_boundary_elms,
        delta_x=delta_x,
        grad_norm=grad_norm,
        diversity_index_1=diversity_index_1,
        diversity_index_2=diversity_index_2,
    )
    return ret_dict
