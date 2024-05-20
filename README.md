# Efficient Diversified Attack (EDA)

## Overview

Python implementation of "Efficient Diversified Attack: Multiple Diversification Strategies Lead to the Efficient Adversarial Attacks".

## Requirements

- Python >= 3.9
- CUDA >= 11.6
- PyTorch >= 1.12.0+cu116
- TorchVision >= 0.13.0+cu116
- gcc >= 4.5

## Installation

Installing dependencies

```bash
pip install -U pip && pip install -r requirements.txt
```

Build C++ extensions

```bash
cd src/extension && python setup.py build_ext --inplace && cd -
```

## Dataset

+ ImageNet
  1. `cd data/imagenet`
  2. Download `ILSVRC2012_img_val.tar` and `ILSVRC2012_devkit_t12.tar.gz` from [ImageNet official site](https://image-net.org/index.php)
  3. `mkdir val && tar -xf ILSVRC2012_img_val.tar -C ./val`
  4. `tar -xzf ILSVRC2012_devkit_t12.tar.gz`
  5. `python build_dataset.py`
  6. `mv val val_original && mv ILSVRC2012_img_val_for_ImageFolder val`

## Usage

### Run EDA

#### Example execution

```bash
python efficient_diversified_attack.py -p ../configs/config_eda.yaml -g 0 -o ../debug --log_level 20 --export_level 60
```

#### Details

```bash
python efficient_diversified_attack.py [-h] -o OUTPUT_DIR -p [PARAM ...]
                                       [--cmd_param [CMD_PARAM ...]] [-g GPU]
                                       [-bs BATCH_SIZE] [--n_threads N_THREADS]
                                       [--image_indices IMAGE_INDICES]
                                       [--log_level LOG_LEVEL]
                                       [--export_level {10,20,30,40,50,60}]
                                       [--experiment]
```

Optional arguments:

```bash
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory name (not path)
  -p [PARAM ...], --param [PARAM ...]
  --cmd_param [CMD_PARAM ...]
                        list of "<param>:<cast type>:<value>" ex) model_name:str:XXX param.max_iter:int:100
  -g GPU, --gpu GPU
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
  --n_threads N_THREADS
  --image_indices IMAGE_INDICES
                        path to yaml file which contains target image indices
  --log_level LOG_LEVEL
                        10:DEBUG,20:INFO,30:WARNING,40:ERROR,50:CRITICAL
  --export_level {10,20,30,40,50,60}
  --experiment          attack all images when this flag is on
```

The default parameter file for EDA is `configs/config_eda.yaml`.

### Run Global search and Local search

#### Example execution

```
python global_search_and_local_search.py -p ../configs/config_gsls.yaml -g 0 -o ../debug --log_level 20 --export_level 60
```

#### Details

```bash
python global_search_and_local_search.py [-h] -o OUTPUT_DIR -p [PARAM ...] [--cmd_param [CMD_PARAM ...]] [-g GPU] [-bs BATCH_SIZE] [--n_threads N_THREADS]
                                         [--image_indices IMAGE_INDICES] [--log_level LOG_LEVEL] [--export_level {10,20,30,40,50,60}] [--experiment]
```

Optional arguments:

```bash
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory name (not path)
  -p [PARAM ...], --param [PARAM ...]
  --cmd_param [CMD_PARAM ...]
                        list of "param:cast type:value" ex) model_name:str:XXX param.max_iter:int:100
  -g GPU, --gpu GPU
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
  --n_threads N_THREADS
  --image_indices IMAGE_INDICES
                        path to yaml file which contains target image indices
  --log_level LOG_LEVEL
                        10:DEBUG,20:INFO,30:WARNING,40:ERROR,50:CRITICAL
  --export_level {10,20,30,40,50,60}
  --experiment          attack all images when this flag is on
```

### Run Target Selection and Targeted attack

#### Example execution

```bash
python target_selection_and_targeted_attack.py -p ../configs/config_tsta.yaml -g 0 -o ../debug --log_level 20 --export_level 60
```

#### Details

```bash
python target_selection_and_targeted_attack.py [-h] -o OUTPUT_DIR -p [PARAM ...] [--cmd_param [CMD_PARAM ...]] [-g GPU] [-bs BATCH_SIZE] [--n_threads N_THREADS]
                                               [--image_indices IMAGE_INDICES] [--log_level LOG_LEVEL] [--export_level {10,20,30,40,50,60}] [--experiment]
```

Optional arguments:

```bash
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory name (not path)
  -p [PARAM ...], --param [PARAM ...]
  --cmd_param [CMD_PARAM ...]
                        list of "param:cast type:value" ex) model_name:str:XXX param.max_iter:int:100
  -g GPU, --gpu GPU
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
  --n_threads N_THREADS
  --image_indices IMAGE_INDICES
                        path to yaml file which contains target image indices
  --log_level LOG_LEVEL
                        10:DEBUG,20:INFO,30:WARNING,40:ERROR,50:CRITICAL
  --export_level {10,20,30,40,50,60}
  --experiment          attack all images when this flag is on
```

## Result of Evaluation

The execution result files are stored under `OUTPUT_DIR` specified with the `-o` option, in a structure like `OUTPUT_DIR/<date>/<time>/<dataset>/<model_name>/`.

The attack success rate and execution time are listed in `OUTPUT_DIR/<date>/<time>/<dataset>/<model_name>/short_summary_<method>.txt`.

## Reproduce the experiments

### Execute GS+LS (Section 4.1)

The execution commands are generated by executing the following command.

```
cd reproduce_experiments/; python generate_cmds_gsls.py 
```

Then,  move to `src` and execute `sh cmds_all_ads_gsls.sh`.

### Execute EDA (Section 4.2)

The execution commands are generated by executing the following command.

```
cd reproduce_experiments/; python generate_cmds_eda.py 
```

Then,  move to `src` and execute `sh cmds_all_eda.sh`.

### Execute EDA and target selection with several initial points

The execution commands are generated by executing the following command.

```
cd reproduce_experiments/; python generate_cmds_eda_and_target_selection.py 
```

Then,  move to `src` and execute `sh cmds_all_eda_ablation.sh`.


