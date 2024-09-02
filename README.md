# Experiment code for ``Efficient Self-supervised Pretraining with Simple Synthesized ECG``

## Prepare directories

- `PATH_TO_ORIGINAL_DATA`: Edit line7 of `invoke_container.sh` 
- `PATH_TO_PROCESSED_DATA_SAVE_DIR`: Edit line8 of `invoke_container.sh`

## Data preparation

1. Real-world data.

- Download CPSC, G12EC, and PTBXL dataset and place at `PATH_TO_ORIGINAL_DATA` 
- Move to `src/prep/dataset` and run `cpsc.py`, `g12ec.py`, and `ptbxl.py`.

2. Synthesized data.

- Move to `src/prep/syn` and run `bash syn_data.sh`


## Baseline selection

- Move to `src/baselines`
- `cd resources` and run `python generate_yaml.py`
- `cd ..` and run `python experiment.py --exp <exp_id>`

## MAE pretraining

- Move to `src/mae_pt`
- `cd resources` and run `python prepare_pt_yamls.py`
- `cd ..` and run `python pretrain.py --pt <pretrain_id>`

## Abnormal ECG classification

- Move to `src/mae_exp`
- `cd resources` and run `python generate_yaml.py`
- `cd ..` and run `python experiment.py --exp <exp_id>`

