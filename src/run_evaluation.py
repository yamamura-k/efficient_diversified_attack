import os
import sys
import time
import torch
import datetime

import yaml


from evaluator_base import EvaluatorBase
from core.criterion import Criterion, CriterionManager

from attacker.attacker_linf import AttackerLinf
from utils import (
    read_yaml,
    overwrite_config,
    set_configurations,
    setup_logger,
    argparser,
    tensor2csv,
    clean_acc,
    reproducibility,
    correct_param,
    load_model_and_dataset
)


class EvaluatorLinf(EvaluatorBase):
    def __init__(self, config, *args, **kwargs):
        super(EvaluatorLinf, self).__init__(config, *args, **kwargs)

    @torch.no_grad()
    def run(
        self,
        model,
        x_test,
        y_test,
        target_image_indices_all,
        target_indices,
        EXPORT_LEVEL=60,
        EXPERIMENT=False,
    ):
        self.postprocess.setLevel(EXPORT_LEVEL)
        today = datetime.date.today().isoformat()
        _time = ":".join(datetime.datetime.now().time().isoformat().split(":")[:2])
        output_dir = os.path.join(self.config.output_dir, today, _time)
        os.makedirs(output_dir, exist_ok=True)

        param = self.config.param.copy()
        param_normalization = self.config.normalization.copy()
        param_stepsize = self.config.stepsize.copy()
        param_algorithm = self.config.algorithm.copy()
        param_initialpoint = self.config.initialpoint.copy()

        dataset = self.config.dataset
        # threat_model = self.config.threat_model

        device = (
            torch.device(f"cuda:{self.config.gpu_id}")
            if torch.cuda.is_available() and self.config.gpu_id is not None
            else torch.device("cpu")
        )
        correct_param(param, param_initialpoint, param_stepsize, dataset)
        self.attacker = AttackerLinf(
            **param,
            param_algorithm=param_algorithm,
            param_normalization=param_normalization,
            param_initialpoint=param_initialpoint,
            param_stepsize=param_stepsize,
            device=device,
            export_level=EXPORT_LEVEL,
        )

        self.target_image_indices_all = target_image_indices_all.clone()
        self.target_indices = target_indices.clone()

        model_name, bs = self.config.model_name, self.config.batch_size

        model = model.to(device)
        model.eval()
        stime = time.time()
        output_root_dir = os.path.join(
            output_dir,
            dataset,
            model_name,
        )
        os.makedirs(output_root_dir, exist_ok=True)

        _criterion = Criterion(model)
        self.criterion = CriterionManager(_criterion)

        self.x_advs_all = x_test.clone()
        acc = torch.ones((len(x_test),), dtype=bool)  # True iff the image is correctly classified.
        # n_targets = 9 if dataset == "cifar10" else 13
        n_targets = 9 if dataset == "cifar10" else 50 if dataset == "cifar100" else 100
        # Remove images which is misclassified
        acc, cw_loss, y_targets = clean_acc(x_test, y_test, bs, model, device, K=n_targets)
        _clean_acc = (acc.sum() / acc.shape[0]) * 100
        if EXPERIMENT:
            acc = torch.ones((len(x_test),), dtype=bool)
        logger.info(f"clean acc: {_clean_acc:.2f}")

        self.best_cw_loss_all = cw_loss.clone()

        n_backward = 0
        n_forward = len(x_test)

        if hasattr(self.config, "optional"):
            settings = zip(
                *[
                    self.config.optional.algorithm_names,
                    self.config.optional.criterion_names,
                    self.config.optional.normalization_types,
                    self.config.optional.max_iters,
                    self.config.optional.initial_stepsizes,
                    self.config.optional.step_strategies,
                    self.config.optional.initialpoints,
                ]
            )
        else:
            settings = [
                [self.config.algorithm_name,
                self.config.criterion_name,
                None,
                param.max_iter,
                None,
                None,
                None,],
            ]
        for order, (
            algorithm_name,
            criterion_name,
            normalization_type,
            max_iter,
            initial_stepsize,
            step_strategy,
            initialpoint,
        ) in enumerate(settings):
            is_targeted_attack = bool("targeted" in criterion_name)
            if normalization_type is not None:
                param_normalization.norm_type = normalization_type
            if max_iter is not None:
                param.max_iter = max_iter
            if initial_stepsize is not None:
                param_stepsize.initial_stepsize = initial_stepsize
            if step_strategy is not None:
                param_stepsize.strategy = step_strategy
            if initialpoint is not None:
                param_initialpoint.method = initialpoint
            output_sub_dir = os.path.join(
                output_root_dir,
                "-".join(
                    (
                        str(order),
                        algorithm_name,
                        criterion_name,
                        param_normalization.norm_type,
                        param_stepsize.strategy,
                        param_initialpoint.method,
                        str(max_iter),
                    )
                ),
            )
            correct_param(param, param_initialpoint, param_stepsize, dataset)
            assert param.max_iter == param_stepsize.max_iter
            self.attacker.updateParameters(
                **param,
                param_algorithm=param_algorithm,
                param_normalization=param_normalization,
                param_initialpoint=param_initialpoint,
                param_stepsize=param_stepsize,
                device=device,
            )
            if hasattr(self.config, "initialpoint_path") and self.config.initialpoint_path:
                x_storage = torch.load(self.config.initialpoint_path).to(device).clone()
            else:
                x_storage = None
            x_best = (
                self.x_advs_all.clone().to(device)
                if param_initialpoint.method == "best"
                else x_storage
                if param_initialpoint.method == "storage"
                else None
            )
            for target in range(n_targets):
                # You should update the dicts.
                y_target = y_targets[:, target] if is_targeted_attack else None
                solution, n_forward, n_backward, accuracy = self.step(
                    x_best=x_best,
                    x_test=x_test,
                    y_test=y_test,
                    acc=acc,
                    max_iter=max_iter,
                    bs=bs,
                    algo_name=algorithm_name,
                    criterion_name=criterion_name,
                    device=device,
                    n_forward=n_forward,
                    n_backward=n_backward,
                    y_target=y_target,
                )
                output_sub_sub_dir = os.path.join(output_sub_dir, str(target))
                os.makedirs(output_sub_sub_dir, exist_ok=True)
                # -------------------------------------------------------------------------------------------------------------
                x_save = solution.x_adv[acc]
                import math
                n_batches = math.ceil(acc.sum().item() / bs)
                for batch_id in range(n_batches):
                    # _mask = torch.zeros_like(acc)
                    # _mask[begin:end]
                    begin = bs * batch_id
                    end = min(10000, bs * (batch_id + 1))
                    x = x_save[begin:end].to(device)
                    out_save = model(x.to(device))
                    torch.save([x, out_save, acc], os.path.join(output_root_dir, f"sample_and_output-{order}-{batch_id}.pth"))
                # x_save = solution.x_adv[:bs][acc[:bs]]
                # out_save = model(x_save.to(device))
                # torch.save([x_save, out_save, acc[:bs]], os.path.join(output_dir, f"sample_and_output-{order}.pth"))
                # -------------------------------------------------------------------------------------------------------------
                # target_label_collection.update(acc, solution.target_class)
                self.postprocess(solution, output_sub_sub_dir)
                if not EXPERIMENT:
                    acc = torch.logical_and(acc, torch.logical_and(accuracy, self.best_cw_loss_all < 1e-3))
                    logger.info(f"ASR: {acc.sum().item()} / {acc.shape[0]}")

                if not is_targeted_attack:
                    break

        run_yaml_path = os.path.join(
            output_root_dir,
            "run.yaml",
        )
        if not os.path.exists(run_yaml_path):
            with open(run_yaml_path, "w") as file:
                yaml.dump(dict(self.config), file)
        if EXPORT_LEVEL < 60:
            save_path = os.path.join(output_root_dir, "best_cw_loss_all.csv")
            tensor2csv(self.best_cw_loss_all, save_path)
        if EXPORT_LEVEL < 20:
            torch.save(
                self.x_advs_all,
                os.path.join(output_root_dir, "adversarial_examples_all.pth"),
            )

        _robust_acc, _, _ = clean_acc(self.x_advs_all, y_test, bs, model, device)

        failed_indices_path = os.path.join(
            output_root_dir,
            "failed_indices.yaml",
        )
        if not os.path.exists(failed_indices_path):
            with open(failed_indices_path, "w") as file:
                yaml.dump({"indices": torch.where(_robust_acc)[0].tolist()}, file)

        robust_acc = 100 * (_robust_acc.sum() / self.config.n_examples)
        attack_success_rate = 100 - robust_acc
        __asr = ((self.best_cw_loss_all >= 0).sum().item() / self.config.n_examples) * 100
        n_forward += len(x_test)
        short_summary_path = os.path.join(output_root_dir, "short_summary.txt")
        msg = f"\ntotal time (sec) = {time.time() - stime:.3f}\nclean acc(%) = {_clean_acc:.2f}\nrobust acc(%) = {robust_acc:.2f}\nASR(%) = {attack_success_rate:.2f}\nASR from cw(%) = {__asr:.2f}\nForward = {n_forward}\nBackward = {n_backward}"
        with open(short_summary_path, "w") as f:
            f.write(msg)
        logger.info(msg)
        if abs(__asr - attack_success_rate) > 1e-3:
            logger.warning(f"ASR: {attack_success_rate:.2f} != ASR(CW): {__asr:.2f}")


@torch.no_grad()
def main(args):
    config = read_yaml(args.param)
    # overwrite by cmd_param
    if args.cmd_param is not None:
        config = overwrite_config(args.cmd_param, config)
    set_configurations(config, args)

    # config["param"]["initialpoint"]["dataset"] = config.dataset

    torch.set_num_threads(args.n_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.n_threads)

    image_indices_yaml = args.image_indices
    target_indices = torch.arange(0, config.n_examples, 1, dtype=int)
    image_indices_all = torch.arange(0, config.n_examples, 1, dtype=int)
    if image_indices_yaml is not None:
        # attack specified images
        target_indices = torch.tensor(read_yaml(image_indices_yaml).indices)

    model, x_test, y_test = load_model_and_dataset(config.model_name, config.dataset, config.n_examples, config.threat_model)
    evaluator = EvaluatorLinf(config)
    evaluator.run(
        model=model,
        x_test=x_test,
        y_test=y_test,
        target_image_indices_all=image_indices_all,
        target_indices=target_indices,
        EXPORT_LEVEL=args.export_level,
        EXPERIMENT=args.experiment,
    )


if __name__ == "__main__":
    # torch.autograd.anomaly_mode.set_detect_anomaly(True)
    sys.path.append("../pyadv")
    # reproducibility()
    parser = argparser()
    args = parser.parse_args()
    logger = setup_logger.setLevel(args.log_level)
    main(args)
