import os
import sys
import time
import torch
import datetime

import yaml


from evaluator_base import EvaluatorBase
from automated_diversified_selection import ADS
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
    load_model_and_dataset,
)

logger = setup_logger(__name__)


class GSLS(EvaluatorBase):
    """ Global search (GS) and Local search (LS)
    """
    def __init__(self, config, *args, **kwargs):
        super(GSLS, self).__init__(config, *args, **kwargs)

    @torch.no_grad()
    def run(
        self,
        model,
        x_test,
        y_test,
        target_image_indices_all,
        target_indices,
        today,
        _time=None,
        EXPORT_LEVEL=60,
        EXPERIMENT=False,
        order=0,
        acc=None,
        cw_loss=None,
        y_targets=None,
        N_targets=9,
    ):
        self.postprocess.setLevel(EXPORT_LEVEL)
        if _time is None:
            _time = ":".join(datetime.datetime.now().time().isoformat().split(":")[:2])
        output_dir = os.path.join(self.config.output_dir, today, _time)
        os.makedirs(output_dir, exist_ok=True)

        param = self.config.param.copy()
        param_normalization = self.config.normalization.copy()
        param_stepsize = self.config.stepsize.copy()
        param_algorithm = self.config.algorithm.copy()
        param_initialpoint = self.config.initialpoint.copy()

        dataset = self.config.dataset
        threat_model = self.config.threat_model

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

        n_backward = 0
        n_forward = 0
        if acc is None or cw_loss is None or y_targets is None:
            acc, cw_loss, y_targets = clean_acc(x_test, y_test, bs, model, device, K=N_targets)
            n_forward += len(x_test)
            ret_acc = acc.clone()
        _clean_acc = (acc.sum() / acc.shape[0]) * 100
        logger.info(f"clean acc: {_clean_acc:.2f}")

        self.best_cw_loss_all = cw_loss.clone()

        ads = ADS(self.config, self.attacker, self.criterion)

        candidates = [
            (a, c)
            for a in ["GD", "APGD", "Nes", "CG"]
            for c in [
                "ce",
                "cw",
                "softmax_cw",
                "g_dlr_1_3",
                "g_dlr_1_4",
                "g_dlr_1_5",
                "g_dlr_1_6",
            ]
        ]
        settings, nfwd, nbwd = ads(
            x_best=None,
            x_test=x_test,
            y_test=y_test,
            acc=acc,
            target_indices=target_indices,
            target_image_indices_all=target_image_indices_all,
            best_cw_loss_all=self.best_cw_loss_all,
            candidates=candidates,
            max_iter=self.config.max_iter,
            stepsize=2 * param.epsilon,
            initial_point=param_initialpoint.method,
            bs=bs,
            device=device,
            sample_ratio=self.config.sample_ratio,
            n_algorithms=self.config.n_algorithms,
            seed=self.config.seed,
            strategy_idx=self.config.ranking_strategy,
        )
        with open(os.path.join(output_root_dir, "selected_attacks_1.txt"), "w") as f:
            for line in settings:
                f.write(f"{line[0]} {line[1]}\n")

        n_forward += nfwd
        n_backward += nbwd

        N = param.max_iter
        initialpoint = param_initialpoint.method

        paramsets = (
            (int(self.config.phi_1 * N), 2 * param.epsilon, initialpoint),
            (int(self.config.phi_2 * N), param.epsilon, "best"),
        )

        for algo_name, criterion_name in settings:
            for (max_iter, stepsize, initp) in paramsets:
                param_normalization.norm_type = "sign"
                param_stepsize.strategy = "static"
                param.max_iter = max_iter
                param_stepsize.initial_stepsize = stepsize
                param_initialpoint.method = initp
                correct_param(param, param_initialpoint, param_stepsize, dataset)
                self.attacker.updateParameters(
                    **param,
                    param_algorithm=param_algorithm,
                    param_normalization=param_normalization,
                    param_initialpoint=param_initialpoint,
                    param_stepsize=param_stepsize,
                    device=device,
                )

                x_best = None if initp == initialpoint else solution.x_adv.clone().to(device)
                solution, n_forward, n_backward, accuracy = self.step(
                    x_best=x_best,
                    x_test=x_test,
                    y_test=y_test,
                    acc=acc,
                    max_iter=max_iter,
                    bs=bs,
                    algo_name=algo_name,
                    criterion_name=criterion_name,
                    device=device,
                    n_forward=n_forward,
                    n_backward=n_backward,
                    y_target=None,
                )
                # target_label_collection.update(acc, solution.target_class)
                if EXPORT_LEVEL < 60:
                    output_sub_dir = os.path.join(
                        output_root_dir,
                        "-".join([str(order), algo_name, criterion_name, str(max_iter)]),
                    )
                    os.makedirs(output_sub_dir, exist_ok=True)
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
                    order += 1
                    self.postprocess(solution, output_sub_dir)

                if not EXPERIMENT:
                    acc = torch.logical_and(acc, torch.logical_and(accuracy, self.best_cw_loss_all < 1e-3))
            # param.max_iter = N
            # param_stepsize.initial_stepsize = 2 * param.epsilon
            # param_initialpoint.method = initialpoint
            # correct_param(param, param_initialpoint, param_stepsize, dataset)
            # self.attacker.updateParameters(
            #     **param,
            #     param_algorithm=param_algorithm,
            #     param_normalization=param_normalization,
            #     param_initialpoint=param_initialpoint,
            #     param_stepsize=param_stepsize,
            #     device=device,
            # )

            # x_best = None
            # solution, n_forward, n_backward, accuracy = self.step(
            #     x_best=x_best,
            #     x_test=x_test,
            #     y_test=y_test,
            #     acc=acc,
            #     max_iter=N,
            #     bs=bs,
            #     algo_name=algo_name,
            #     criterion_name=criterion_name,
            #     device=device,
            #     n_forward=n_forward,
            #     n_backward=n_backward,
            #     y_target=None,
            # )
            # # target_label_collection.update(acc, solution.target_class)
            # if EXPORT_LEVEL < 60:
            #     output_sub_dir = os.path.join(
            #         output_root_dir,
            #         "-".join([str(order), algo_name, criterion_name, str(N)]),
            #     )
            #     os.makedirs(output_sub_dir, exist_ok=True)
            #     # -------------------------------------------------------------------------------------------------------------
            #     torch.save([solution.x_adv[acc], acc], os.path.join(output_dir, f"sample_and_output-{order}.pth"))
            #     # -------------------------------------------------------------------------------------------------------------
            #     order += 1
            #     self.postprocess(solution, output_sub_dir)

            # if not EXPERIMENT:
            #     acc = torch.logical_and(acc, torch.logical_and(accuracy, self.best_cw_loss_all < 1e-3))

        if not self.config.single:
            prev = self.best_cw_loss_all.clone()  # TODO
            settings, nfwd, nbwd = ads(
                x_best=self.x_advs_all.clone().to(device),
                x_test=x_test,
                y_test=y_test,
                acc=acc,
                target_indices=target_indices,
                target_image_indices_all=target_image_indices_all,
                best_cw_loss_all=self.best_cw_loss_all,
                candidates=candidates,
                max_iter=self.config.max_iter,
                stepsize=param.epsilon / 2,
                initial_point="best",
                bs=bs,
                device=device,
                sample_ratio=self.config.sample_ratio,
                n_algorithms=self.config.n_algorithms,
                seed=self.config.seed,
                strategy_idx=self.config.ranking_strategy,
            )
            with open(os.path.join(output_root_dir, "selected_attacks_2.txt"), "w") as f:
                for line in settings:
                    f.write(f"{line[0]} {line[1]}\n")
            n_forward += nfwd
            n_backward += nbwd

        for algo_name, criterion_name in settings:
            max_iter = int((1 - self.config.phi_1 - self.config.phi_2) * N)
            param_normalization.norm_type = "sign"
            param.max_iter = max_iter
            param_initialpoint.method = "best"
            param_stepsize.initial_stepsize = param.epsilon / 2
            if algo_name == "CG":
                param_stepsize.strategy = "apgd"
            else:
                param_stepsize.strategy = "cos"
            correct_param(param, param_initialpoint, param_stepsize, dataset)
            self.attacker.updateParameters(
                **param,
                param_algorithm=param_algorithm,
                param_normalization=param_normalization,
                param_initialpoint=param_initialpoint,
                param_stepsize=param_stepsize,
                device=device,
            )

            solution, n_forward, n_backward, accuracy = self.step(
                x_best=self.x_advs_all.clone().to(device),
                x_test=x_test,
                y_test=y_test,
                acc=acc,
                max_iter=max_iter,
                bs=bs,
                algo_name=algo_name,
                criterion_name=criterion_name,
                device=device,
                n_forward=n_forward,
                n_backward=n_backward,
                y_target=None,
            )
            # target_label_collection.update(acc, solution.target_class)
            if EXPORT_LEVEL < 60:
                output_sub_dir = os.path.join(
                    output_root_dir,
                    "-".join([str(order), algo_name, criterion_name, str(max_iter)]),
                )
                os.makedirs(output_sub_dir, exist_ok=True)
                # -------------------------------------------------------------------------------------------------------------
                x_save = solution.x_adv[acc]
                # out_save = model(x_save.to(device))
                # torch.save([x_save, out_save, acc], os.path.join(output_dir, f"sample_and_output-{order}.pth"))
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
                order += 1
                self.postprocess(solution, output_sub_dir)

            exploitation_inds = torch.logical_and(
                torch.logical_not(
                    torch.logical_or(
                        torch.logical_not(acc),
                        solution.best_softmax_cw_loss[:, -1] >= 1e-3,
                    )
                ),
                solution.best_softmax_cw_loss[:, -1] >= -5e-2,
            )
            if exploitation_inds.any():
                param_stepsize.initial_stepsize = param.epsilon / 4
                correct_param(param, param_initialpoint, param_stepsize, dataset)
                self.attacker.updateParameters(
                    **param,
                    param_algorithm=param_algorithm,
                    param_normalization=param_normalization,
                    param_initialpoint=param_initialpoint,
                    param_stepsize=param_stepsize,
                    device=device,
                )
                exploitation_criterion_name = criterion_name.split("-")[-1]
                solution, n_forward, n_backward, accuracy = self.step(
                    x_best=self.x_advs_all.clone().to(device),
                    x_test=x_test,
                    y_test=y_test,
                    acc=exploitation_inds,
                    max_iter=max_iter,
                    bs=bs,
                    algo_name=algo_name,
                    criterion_name=exploitation_criterion_name,
                    device=device,
                    n_forward=n_forward,
                    n_backward=n_backward,
                    y_target=None,
                )
                if EXPORT_LEVEL < 60:
                    output_sub_dir_2 = os.path.join(
                        output_root_dir,
                        "-".join(
                            [
                                str(order),
                                "exploitation",
                                algo_name,
                                criterion_name,
                                str(max_iter),
                            ]
                        ),
                    )
                    # -------------------------------------------------------------------------------------------------------------
                    x_save = solution.x_adv[acc]
                    n_batches = math.ceil(acc.sum().item() / bs)
                    for batch_id in range(n_batches):
                        # _mask = torch.zeros_like(acc)
                        # _mask[begin:end]
                        begin = bs * batch_id
                        end = min(10000, bs * (batch_id + 1))
                        x = x_save[begin:end].to(device)
                        out_save = model(x.to(device))
                        torch.save([x, out_save, acc], os.path.join(output_root_dir, f"sample_and_output-{order}-{batch_id}.pth"))
                    # out_save = model(x_save.to(device))
                    # torch.save([x_save, out_save, acc], os.path.join(output_dir, f"sample_and_output-{order}.pth"))
                    # x_save = solution.x_adv[:bs][acc[:bs]]
                    # out_save = model(x_save.to(device))
                    # torch.save([x_save, out_save, acc[:bs]], os.path.join(output_dir, f"sample_and_output-{order}.pth"))
                    # -------------------------------------------------------------------------------------------------------------
                    os.makedirs(output_sub_dir_2, exist_ok=True)
                    order += 1
                    self.postprocess(solution, output_sub_dir_2)

            if not EXPERIMENT:
                acc = torch.logical_and(acc, torch.logical_and(accuracy, self.best_cw_loss_all < 1e-3))

        run_yaml_path = os.path.join(
            output_root_dir,
            "run.yaml",
        )
        if not os.path.exists(run_yaml_path):
            with open(run_yaml_path, "w") as file:
                yaml.dump(dict(self.config), file)
        if EXPORT_LEVEL < 60:
            save_path = os.path.join(output_root_dir, "best_cw_loss_all_gsls.csv")
            tensor2csv(self.best_cw_loss_all, save_path)
        if EXPORT_LEVEL < 20:
            torch.save(
                self.x_advs_all,
                os.path.join(output_root_dir, "adversarial_examples_all_gsls.pth"),
            )
        _robust_acc, _, _ = clean_acc(self.x_advs_all, y_test, bs, model, device)

        failed_indices_path = os.path.join(
            output_root_dir,
            "failed_indices_gsls.yaml",
        )
        if not os.path.exists(failed_indices_path):
            with open(failed_indices_path, "w") as file:
                yaml.dump({"indices": torch.where(_robust_acc)[0].tolist()}, file)

        robust_acc = 100 * (_robust_acc.sum() / self.config.n_examples)
        attack_success_rate = 100 - robust_acc
        __asr = ((self.best_cw_loss_all >= 0).sum().item() / self.config.n_examples) * 100
        n_forward += len(x_test)
        short_summary_path = os.path.join(output_root_dir, "short_summary_gsls.txt")
        msg = f"\ntotal time (sec) = {time.time() - stime:.3f}\nclean acc(%) = {_clean_acc:.2f}\nrobust acc(%) = {robust_acc:.2f}\nASR(%) = {attack_success_rate:.2f}\nASR from cw(%) = {__asr:.2f}\nForward = {n_forward}\nBackward = {n_backward}"
        with open(short_summary_path, "w") as f:
            f.write(msg)
        logger.info(msg)
        return (
            self.x_advs_all,
            self.best_cw_loss_all,
            _robust_acc,
            order,
            n_forward,
            n_backward,
            ret_acc,
            cw_loss,
            y_targets,
        )


@torch.no_grad()
def main(args):
    config = read_yaml(args.param)
    # overwrite by cmd_param
    if args.cmd_param is not None:
        config = overwrite_config(args.cmd_param, config)
    set_configurations(config, args)

    torch.set_num_threads(args.n_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.n_threads)

    image_indices_yaml = args.image_indices
    target_indices = torch.arange(0, config.n_examples, 1, dtype=int)
    image_indices_all = torch.arange(0, config.n_examples, 1, dtype=int)
    if image_indices_yaml is not None:
        # attack specified images
        target_indices = torch.tensor(read_yaml(image_indices_yaml).indices)
    model, x_test, y_test = load_model_and_dataset(config.model_name, config.dataset, config.n_examples, config.threat_model)

    today = datetime.date.today().isoformat()
    evaluator = GSLS(config)
    evaluator.run(
        model=model,
        x_test=x_test,
        y_test=y_test,
        target_image_indices_all=image_indices_all,
        target_indices=target_indices,
        EXPORT_LEVEL=args.export_level,
        EXPERIMENT=args.experiment,
        today=today,
    )


if __name__ == "__main__":
    # torch.autograd.anomaly_mode.set_detect_anomaly(True)
    sys.path.append("../src")
    reproducibility()
    parser = argparser()
    args = parser.parse_args()
    logger = setup_logger.setLevel(args.log_level)
    main(args)
