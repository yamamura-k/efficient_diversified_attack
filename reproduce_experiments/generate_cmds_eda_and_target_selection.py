import os
import glob
import json
import yaml
import datetime
use_models = {
    ("Addepalli2022Efficient_RN18", "cifar10"),
    ("Addepalli2022Efficient_WRN_34_10", "cifar10"),
    ("Andriushchenko2020Understanding", "cifar10"),
    ("Carmon2019Unlabeled", "cifar10"),
    ("Cui2020Learnable_34_10", "cifar10"),
    ("Cui2020Learnable_34_20", "cifar10"),
    ("Ding2020MMA", "cifar10"),
    ("Engstrom2019Robustness", "cifar10"),
    ("Gowal2020Uncovering_28_10_extra", "cifar10"),
    ("Gowal2020Uncovering_34_20", "cifar10"),
    ("Gowal2020Uncovering_70_16_extra", "cifar10"),
    ("Gowal2020Uncovering_70_16", "cifar10"),
    ("Hendrycks2019Using", "cifar10"),
    ("Huang2020Self", "cifar10"),
    ("Pang2020Boosting", "cifar10"),
    ("Sehwag2021Proxy", "cifar10"),
    ("Sitawarin2020Improving", "cifar10"),
    ("Sridhar2021Robust_34_15", "cifar10"),
    ("Sridhar2021Robust", "cifar10"),
    ("Wang2020Improving", "cifar10"),
    ("Wu2020Adversarial_extra", "cifar10"),
    ("Zhang2019Theoretically", "cifar10"),
    ("Zhang2019You", "cifar10"),
    ("Zhang2020Attacks", "cifar10"),
    ("Rebuffi2021Fixing_28_10_cutmix_ddpm", "cifar10"),
    ("Rebuffi2021Fixing_70_16_cutmix_ddpm", "cifar10"),
    ("Rice2020Overfitting", "cifar10"),
    ("Sehwag2020Hydra", "cifar10"),
    ("Sehwag2021Proxy_R18", "cifar10"),

    ("Addepalli2022Efficient_WRN_34_10", "cifar100"),
    ("Cui2020Learnable_34_10_LBGAT6", "cifar100"),
    ("Cui2020Learnable_34_20_LBGAT6", "cifar100"),
    ("Gowal2020Uncovering", "cifar100"),
    ("Gowal2020Uncovering_extra", "cifar100"),
    ("Hendrycks2019Using", "cifar100"),
    ("Rebuffi2021Fixing_28_10_cutmix_ddpm", "cifar100"),
    ("Rebuffi2021Fixing_70_16_cutmix_ddpm", "cifar100"),
    ("Rice2020Overfitting", "cifar100"),
    ("Sitawarin2020Improving", "cifar100"),
    ("Wu2020Adversarial", "cifar100"),

    ("Salman2020Do_50_2", "imagenet"),
    ("Salman2020Do_R18", "imagenet"),
    ("Salman2020Do_R50", "imagenet"),
    ("Wong2020Fast", "imagenet"),
    ("Engstrom2019Robustness", "imagenet"),
}

identify_words = "eda_ablation"
log_level = 60
export_level = 60
cmd_1 = [[] for _ in range(4)]
n_gpu = 1
s_gpu = 0
idx = 0
seed = 0

max_iter = 100
# initial_point = "pas"
additional = True
factor = 1 # 1 # 1.3 # 3
ranking_strategy = 2
for initial_point in ["input", "odi", "pas"]:
    optional = f"{initial_point}-{initial_point}-best-strategy_{ranking_strategy}"


    f = open("../model_info.yaml", "r")
    all_models = yaml.safe_load(f)
    f.close()
    base_cmd_1 = f"python -B efficient_diversified_attack.py -p ../configs/config_eda.yaml --log_level {log_level} --export_level {export_level}"
    base_cmd_2 = f"python -B target_selection_and_targeted_attack.py -p ../configs/config_eda.yaml --log_level {log_level} --export_level {export_level}"

    for dataset in ["cifar10", "cifar100", "imagenet"]:
        with open(f"../exp/{dataset}_rtx3090.json", "r") as f:
            data = json.load(f)
        n_examples = 5000 if dataset == "imagenet" else 10000
        for model_name, bs in sorted(data.items(), key=lambda x: x[1], reverse=True):
            model_name = model_name.split("/")[-1].split(".")[0]
            if (model_name, dataset) not in use_models:
                continue
            epsilon = all_models[dataset][model_name]["eps"]
            out_dir = f"../result/{optional}"
            cmd_1[idx].append(
                f"{base_cmd_1} -g {s_gpu+(idx%4)} -o {out_dir} --n_threads 10 --cmd_param n_examples:int:{n_examples} model_name:str:{model_name} batch_size:int:{int(bs * factor)} dataset:str:{dataset} param.epsilon:float:{epsilon} stepsize.initial_stepsize:float:{2 * epsilon} param.max_iter:int:{max_iter} initialpoint.method:str:{initial_point} additional:bool:{additional} seed:int:{seed} ranking_strategy:int:{ranking_strategy}"
            )
            cmd_1[idx].append(
                f"{base_cmd_2} -g {s_gpu+(idx%4)} -o {out_dir} --n_threads 10 --cmd_param n_examples:int:{n_examples} model_name:str:{model_name} batch_size:int:{int(bs * factor)} dataset:str:{dataset} param.epsilon:float:{epsilon} stepsize.initial_stepsize:float:{2 * epsilon} param.max_iter:int:{max_iter} initialpoint.method:str:{initial_point} additional:bool:{additional} seed:int:{seed} ranking_strategy:int:{ranking_strategy}"
            )
            idx += 1
            idx %= n_gpu


source_paths = ["../src"]
cmds_all = []
for i in range(n_gpu):
    filename = f"{datetime.datetime.today().date().isoformat()}_{identify_words}_{i+s_gpu}.sh"
    cmds_all.append("sh ./" + filename + " &")
    f = open(
        os.path.join(*source_paths, filename), "w"
    )
    f.write("\n".join(cmd_1[i]))
    f.close()
f = open(os.path.join(*source_paths, f"cmds_all_{identify_words}.sh"), "w")
f.write("\n".join(cmds_all))
f.close()
