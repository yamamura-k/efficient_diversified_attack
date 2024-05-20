import math
import torch
from .args import argparser
from .configuration import (
    set_configurations,
    get_machine_info,
    reproducibility,
    correct_param,
)
from .logging import setup_logger
from .loader import load_imagenet, load_model_and_dataset
from .io import (
    InformationWriter,
    read_yaml,
    overwrite_config,
    tensor2csv,
    compute_information,
)

DEBUG = False


def updateParam(obj, new_param):
    for key in new_param:
        setattr(obj, key, new_param[key])


def h(k, K, ymin=0.01, alpha=14):
    return torch.round(K * (torch.exp(-alpha * (k + 0.05 * K) / K) + ymin)).to(torch.int)


@torch.inference_mode()
def clean_acc(x, y, batch_size, model, device, K=9):
    n_examples = len(x)
    acc = torch.ones((n_examples,), dtype=bool)
    cw_loss = torch.ones((n_examples,))
    nbatches = math.ceil(n_examples / batch_size)
    target_image_indices_all = torch.arange(n_examples, dtype=torch.long)
    y_target = torch.ones((n_examples, K), dtype=torch.long)

    for idx in range(nbatches):
        begin = idx * batch_size
        end = min((idx + 1) * batch_size, n_examples)
        target_image_indices = target_image_indices_all[begin:end]
        logit = model(x[target_image_indices].clone().to(device)).cpu()
        preds = logit.argmax(1)
        acc[target_image_indices] = preds == y[target_image_indices]
        inds = logit.argsort(1)
        cw_loss[target_image_indices] = (
            logit[torch.arange(end - begin), inds[:, -2]] * acc[target_image_indices].float()
            + logit[torch.arange(end - begin), inds[:, -1]] * (1.0 - acc[target_image_indices].float())
            - logit[torch.arange(end - begin), y[target_image_indices]]
        )
        for _k in range(2, K + 2):
            y_target[target_image_indices, _k - 2] = inds[:, -_k]
            _inds = (y_target[target_image_indices, _k - 2] == y[target_image_indices]).to(torch.bool)
            y_target[target_image_indices, _k - 2][_inds] = inds[_inds][:, -_k]
    return acc, cw_loss, y_target
