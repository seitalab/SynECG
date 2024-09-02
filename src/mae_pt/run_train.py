import os
from glob import glob
from typing import Tuple, Optional
from argparse import Namespace

import torch
import numpy as np
from optuna.trial import Trial

from codes.supports import utils
from codes.pretrain_model import ModelPretrainer
from codes.finetune_model import ModelFinetuner

torch.backends.cudnn.deterministic = True

def update_args(args):

    args.dataset = None

    return args

def find_weight_file(dirname: str, seed: int):
    """
    Form path to `.pth` file.
    If `net.pth` file does not exist (MAE -> Syn clf -> finetune), 
    then search for path and fix path.

    Args:
        target_loc (str): 
        seed (int): _description_
    """
    weight_file = os.path.join(dirname, "net.pth")
    if os.path.exists(weight_file):
        return weight_file, False
    dirname_ext = os.path.join(
        dirname, 
        "multirun/train",
        f"seed{seed:04d}",
    )
    weight_file = glob(dirname_ext + "/*/net.pth")
    assert len(weight_file) == 1
    return weight_file[0], True

def run_train(
    args: Namespace, 
    save_root: str,
    trial: Optional[Trial]=None,
    finetune_target: Optional[str]=None
) -> Tuple[float, str]:
    """
    Execute train code for ecg classifier
    Args:
        args (Namespace): Namespace for parameters used.
        save_root (str): 
    Returns:
        best_val_loss (float): 
        save_dir (str):
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Prepare result storing directories
    timestamp = utils.get_timestamp()
    save_setting = f"{timestamp}-{args.host}"
    save_dir = os.path.join(
        save_root, 
        save_setting
    )

    # Trainer prep
    if finetune_target is not None:
        args = update_args(args)
        trainer = ModelFinetuner(args, save_dir)
        trainer.set_trial(trial)
        trainer.set_model()
        # weight_file = os.path.join(finetune_target, "net.pth")
        weight_file, double_ft = find_weight_file(
            finetune_target, args.seed)

        trainer.set_pretrained_mae(
            weight_file, args.freeze, double_ft=double_ft)
        is_finetune = True
    else:
        trainer = ModelPretrainer(args, save_dir)
        trainer.set_trial(trial)
        trainer.set_model()
        is_finetune = False

    print("Preparing dataloader ...")
    train_loader = trainer.prepare_dataloader(
        "train", is_train=True, is_finetune=is_finetune)
    valid_loader = trainer.prepare_dataloader(
        "val", is_train=False, is_finetune=is_finetune)

    if is_finetune:
        weight = utils.calc_class_weight(train_loader.dataset.label)
        trainer.set_lossfunc(weight)

    trainer.set_optimizer()
    trainer.save_params()

    print("Starting training ...")
    trainer.run(train_loader, valid_loader)
    _, best_result = trainer.get_best_loss()

    del trainer

    # Return best validation loss when executing hyperparameter search.
    return best_result, save_dir

if __name__ == "__main__":

    import socket
    from argparse import ArgumentParser

    import yaml
    
    def prepare_params(args):

        args.host = socket.gethostname()
        args = vars(args)

        cfg_file = args["config"]

        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)
        for param, param_val in cfg.items():
            args[param] = param_val["param_val"]

        args = Namespace(**args)
        return args

    parser = ArgumentParser()
    parser.add_argument(
        '--config', 
        help='path to config file', 
        default="./resources/exp_yamls/exp9999.yaml"
    )
    parser.add_argument('--cpu', action="store_true")
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--seed', default=1)
    parser.add_argument('--eval', default="ptbxl")
    args = parser.parse_args()

    save_root = "/home/nonaka/git/SynthesizedECG/tmp/mae"
    train_params = prepare_params(args)

    # _, save_dir = run_train(train_params, save_root)
    save_dir = "/home/nonaka/git/SynthesizedECG/tmp/mae/231117-174528-kakegawa"
    print(save_dir)

    clf_save_root = save_root + "/../clf"
    _, save_dir = run_train(
        train_params, clf_save_root, finetune_target=save_dir)

    from run_eval import run_eval
    dump_loc = save_root + "/../evaled"
    run_eval(
        save_dir, 
        args.device, 
        dump_loc, 
        eval_dataset=args.eval,
        dump_errors=False
    )