import os
import sys
# import git
import subprocess
import platform

import torch
import random
import numpy as np
from .logging import setup_logger

logger = setup_logger(__name__)


def reproducibility():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_machine_info():
    system = platform.system()
    plf = platform.platform(aliased=True)
    node = platform.node()
    processor = platform.processor()

    return dict(system=system, platform=plf, node=node, processor=processor)


def set_configurations(config, args):
    # add other attributes.
    setattr(config, "output_dir", args.output_dir)
    setattr(config, "gpu_id", args.gpu)
    setattr(config, "yaml_paths", args.param)
    if args.batch_size is not None:
        setattr(config, "batch_size", args.batch_size)

    # execution command
    cmd_argv = " ".join((["python"] + sys.argv))
    setattr(config, "cmd", cmd_argv)

    # environmental variables
    environ_variables = dict(os.environ)
    setattr(config, "environ_variables", environ_variables)

    # git hash
    # git_hash = git.cmd.Git("./").rev_parse("HEAD")
    # setattr(config, "git_hash", git_hash)

    # # branch
    # _cmd = "git rev-parse --abbrev-ref HEAD"
    # branch = subprocess.check_output(_cmd.split()).strip().decode("utf-8")
    # branch = "-".join(branch.split("/"))
    # setattr(config, "branch", branch)

    # machine information
    machine = get_machine_info()
    setattr(config, "machine", machine)


def correct_param(param, param_initialpoint, param_stepsize, dataset):
    param_stepsize["max_iter"] = param["max_iter"]
    param_stepsize["epsilon"] = param["epsilon"]
    param_initialpoint["epsilon"] = param["epsilon"]
    param_initialpoint["dataset"] = dataset
