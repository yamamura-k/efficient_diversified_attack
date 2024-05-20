import os
import sys
import time
import torch
import datetime


from global_search_and_local_search import GSLS
from target_selection_and_targeted_attack import TSTA
from utils import (
    read_yaml,
    overwrite_config,
    set_configurations,
    setup_logger,
    argparser,
    reproducibility,
    load_model_and_dataset,
)


@torch.no_grad()
def main(args):
    stime = time.time()
    config = read_yaml(args.param)
    # overwrite by cmd_param
    if args.cmd_param is not None:
        config = overwrite_config(args.cmd_param, config)
    set_configurations(config, args)
    torch.set_num_threads(args.n_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.n_threads)

    if config.additional:
        N_targets = 9 if config.dataset == "cifar10" else 14 if config.dataset == "cifar100" else 20
    else:
        N_targets = 9

    image_indices_yaml = args.image_indices
    target_indices = torch.arange(0, config.n_examples, 1, dtype=int)
    image_indices_all = torch.arange(0, config.n_examples, 1, dtype=int)
    if image_indices_yaml is not None:
        # attack specified images
        target_indices = torch.tensor(read_yaml(image_indices_yaml).indices)
    model, x_test, y_test = load_model_and_dataset(config.model_name, config.dataset, config.n_examples, config.threat_model)
    today = datetime.date.today().isoformat()
    _time = ":".join(datetime.datetime.now().time().isoformat().split(":")[:2])
    n_forward = 0
    n_backward = 0

    gsls = GSLS(config)
    (x_advs, best_cw, robust_acc, order, n_fwd, n_bwd, acc, cw_loss, y_targets,) = gsls.run(
        model=model,
        x_test=x_test,
        y_test=y_test,
        target_image_indices_all=image_indices_all,
        target_indices=target_indices,
        EXPORT_LEVEL=args.export_level,
        EXPERIMENT=args.experiment,
        today=today,
        _time=_time,
        N_targets=N_targets,
    )
    n_forward += n_fwd
    n_backward += n_bwd
    target_indices_2 = torch.where(robust_acc)[0]
    tsta = TSTA(config)
    (x_advs, best_cw, _robust_acc, order, n_fwd, n_bwd, acc, cw_loss, y_targets,) = tsta.run(
        model=model,
        x_test=x_test,
        y_test=y_test,
        target_image_indices_all=image_indices_all,
        target_indices=target_indices_2,
        EXPORT_LEVEL=args.export_level,
        EXPERIMENT=args.experiment,
        today=today,
        _time=_time,
        order=order,
        acc=acc.clone(),
        cw_loss=cw_loss,
        y_targets=y_targets,
    )
    n_forward += n_fwd
    n_backward += n_bwd
    robust_acc = (torch.logical_and(robust_acc, _robust_acc).sum() / config.n_examples) * 100
    output_root_dir = os.path.join(
        config.output_dir,
        today,
        _time,
        config.dataset,
        config.model_name,
    )
    short_summary_path = os.path.join(output_root_dir, "short_summary_eda.txt")
    msg = f"\ntotal time (sec) = {time.time() - stime:.3f}\nrobust acc(%) = {robust_acc:.2f}\nASR(%) = {100 - robust_acc:.2f}\nForward = {n_forward}\nBackward = {n_backward}"
    with open(short_summary_path, "w") as f:
        f.write(msg)
    logger.info(msg)


if __name__ == "__main__":
    # torch.autograd.anomaly_mode.set_detect_anomaly(True)
    sys.path.append("../src")
    reproducibility()
    parser = argparser()
    args = parser.parse_args()
    logger = setup_logger.setLevel(args.log_level)
    main(args)
