import os
from typing import Tuple, Optional
from argparse import Namespace

import yaml
import torch
import numpy as np
from optuna.trial import Trial

from codes.supports import utils
from codes.train_model import ModelTrainer

torch.backends.cudnn.deterministic = True

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["experiment"]

def update_args(args):

    args.dataset = None

    return args


def get_weight_file_path(finetune_target_id: str):
    """
    Args:
        finetune_target_id (str): <ARCH>-<PriorTraining>_<ID>
    Returns:
        path_to_weight_file (str): 
    """
    return config["mae_settings"]["pt_model_path"][finetune_target_id]

    # target_arch = finetune_target_id.split("-")[0]

    # return config["ft_settings"]["model_path"][target_arch][finetune_target_id]
    
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
    save_dir = os.path.join(save_root, save_setting)

    # Trainer prep
    trainer = ModelTrainer(args, save_dir)
    trainer.set_trial(trial)
    trainer.set_model()
    if finetune_target is not None:
        weight_dir = get_weight_file_path(finetune_target)
        weight_file = os.path.join(weight_dir, "net.pth")
        trainer.set_pretrained_mae(weight_file, args.freeze)

    print("Preparing dataloader ...")
    train_loader = trainer.prepare_dataloader("train", is_train=True)
    valid_loader = trainer.prepare_dataloader("val", is_train=False)

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

    pass