import os
import socket
from glob import glob
from argparse import Namespace

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    average_precision_score,
    roc_auc_score,
    accuracy_score
)

from run_train import run_train
from experiment import ExperimentManager
from codes.supports.utils import get_timestamp

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class PretrainingEvaluationManager(ExperimentManager):

    def __init__(
        self, 
        target_id: int, 
        device: str, 
        use_cpu: bool=False,
        debug: bool=False
    ):
        eval_config_file = os.path.join(
            config["experiment"]["path"]["mae_eval_yaml_loc"],
            f"m{target_id//100:02d}s",
            f"check{target_id:04d}.yaml"
        )

        fixed_params, _, _ =\
            self._load_train_params(eval_config_file)
        fixed_params = self._update_fixed_params(fixed_params)
        self.fixed_params = Namespace(**fixed_params)

        self.fixed_params.seed = config["experiment"]["seed"]["hps"]
        self.fixed_params.host = socket.gethostname()
        self.fixed_params.device = device

        self._prepare_save_loc(target_id)
        self._save_config(eval_config_file)
        self._select_device(device, use_cpu)

        # For debugging.
        if debug:
            self.fixed_params.epochs = 2
            self.fixed_params.eval_every = 1
        else:
            self._prepare_reporter(eval_config_file)
        self.debug = debug

    def _update_fixed_params(self, fixed_params):

        key = fixed_params["pretrain_key"]
        fixed_params.update(config["pretrain_params"]["mae"][key])

        # str -> float
        fixed_params["learning_rate"] = self._str_to_number(
            fixed_params["learning_rate"], to_int=False)

        fixed_params = self._merge_from_mae_setting(fixed_params)
        return fixed_params
    
    def _merge_from_mae_setting(self, fixed_params):
        """
        Load yaml file used during mae pretraining.

        Args:
            fixed_params (dict): 
        Returns:
            fixed_params (dict): 
        """        

        finetune_target = fixed_params["finetune_target"]
        yaml_file = glob(finetune_target + "/../pt????.yaml")
        assert len(yaml_file) == 1

        with open(yaml_file[0]) as f:
            mae_cfg = yaml.safe_load(f)

        for target_param in fixed_params["reuse_params"]:
            fixed_params[target_param] =\
                mae_cfg[target_param]["param_val"]

        del fixed_params["reuse_params"]
        return fixed_params

    def _str_to_number(self, str_num: str, to_int: bool=True):
        """
        Args:
            str_num (str): `XX*1eY`
        """
        str_num = str_num.split("*")
        number = float(str_num[0]) * float(str_num[1])

        if to_int:
            number = int(number)
        return number

    def _prepare_save_loc(self, pretrain_id: int):
        """
        Args:

        Returns:
            None
        """
        self.save_loc = os.path.join(
            config["experiment"]["path"]["save_root"],
            f"mae-eval{pretrain_id:04d}"[:-2]+"s",
            f"check{pretrain_id:04d}",
            get_timestamp()
        )
        os.makedirs(self.save_loc, exist_ok=True)

    def _form_result_csv(self, y_true, y_pred):
        """
        Args:
            None
        Returns:
            None
        """
        y_pred = sigmoid(y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred > 0.5).ravel()
        f1score = f1_score(y_true, y_pred>0.5)
        accuracy = accuracy_score(y_true, y_pred>0.5)
        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)

        result_dict = {
            "f1score": [f1score],
            "accuracy": [accuracy],
            "AUROC": [auroc],
            "AUPRC": [auprc],
            "TP": [tp],
            "FP": [fp],
            "FN": [fn],
            "TN": [tn],            
        }
        df = pd.DataFrame.from_dict(result_dict)
        return df

    def main(self, single_run=False):
        """
        Args:
            None
        Returns:
            None
        """
        result_dict, save_dir = run_train(
            self.fixed_params, 
            self.save_loc,
            finetune_target=self.fixed_params.finetune_target
        )
        result_df = self._form_result_csv(
            result_dict["y_trues"][:, 0], 
            result_dict["y_preds"][:, 0]
        )
        savename = os.path.join(save_dir, "..", "result.csv")
        result_df.to_csv(savename)
        
if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        '--eval', 
        default=1
    )
    parser.add_argument(
        '--device', 
        default="cuda:0"
    )
    parser.add_argument(
        '--debug', 
        action="store_true"
    )
    args = parser.parse_args()

    print(args)

    executer = PretrainingEvaluationManager(
        int(args.eval), 
        args.device,
        debug=args.debug
    )
    executer.main()
