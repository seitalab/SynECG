import os
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

from run_hps import run_hps
from run_train import run_train
from codes.supports.utils import get_timestamp
from codes.supports.param_utils import ParameterManager

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)
proc_data_root = config["path"]["processed_data"]
config = config["experiment"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ExperimentManager:

    def __init__(
        self, 
        exp_id: int, 
        device: str,
        use_cpu: bool=False,
        debug: bool=False
    ):
        """

        Args:
            config_file (str): _description_
            device (str): cuda device or cpu to use.
            use_cpu (bool, optional): _description_. Defaults to False.
            debug (bool, optional): _description_. Defaults to False.
        """

        exp_config_file = os.path.join(
            config["path"]["yaml_loc"],
            f"exp{exp_id:04d}.yaml"
        )

        # Load parameters.
        fixed_params, self.hps_mode, search_params =\
            self._load_train_params(exp_config_file)
        self.fixed_params = Namespace(**fixed_params)
        self.search_params = Namespace(**search_params)

        self._prepare_save_loc(exp_id)
        self._save_config(exp_config_file)
        self._select_device(device, use_cpu)

        # For debugging.
        if debug:
            self.fixed_params.epochs = 2
            self.fixed_params.eval_every = 1
        self.debug = debug

    def _select_device(self, device: str, use_cpu: bool):
        """
        Args:

        Returns:
            None
        """
        if use_cpu:
            device = "cpu"
        self.device = device

    def _prepare_save_loc(self, exp_id: int):
        """
        Args:

        Returns:
            None
        """
        self.save_loc = os.path.join(
            proc_data_root,
            config["path"]["save_root"],
            f"baseline{exp_id//100:02d}s",
            f"baseline{exp_id:04d}",
            get_timestamp()
        )
        os.makedirs(self.save_loc, exist_ok=True)

    def _save_config(self, config_file: str):
        """

        Args:
            config_file (str): _description_
        Returns:
            None
        """ 
        # Copy config file for current execution.
        config_basename = os.path.basename(config_file)
        stored_config_file = os.path.join(self.save_loc, config_basename)
        command = f"cp {config_file} {stored_config_file}"
        os.system(command)

    def _load_train_params(self, config_file: str):
        """

        Args:
            config_file (str): _description_
        Returns:
            fix_params (Dict): 
            hps_mode (bool): True if hps False if grid search.
            search_params (Dict): hps_params or gs_params.
        """ 
        with open(config_file) as f:
            params = yaml.safe_load(f)

        fixed_params, hps_params, gs_params = {}, {}, {}
        for key, value in params.items():
            if type(value) != dict:
                continue

            if value["param_type"] == "fixed":
                fixed_params[key] = value["param_val"]
            elif value["param_type"] == "grid":
                assert type(value["param_val"]) == list
                gs_params[key] = value["param_val"] # List stored
            elif value["param_type"] == "hps":
                assert type(value["param_val"]) == list
                hps_params[key] = value["param_val"]
            else:
                raise NotImplementedError
        # hps_params and gs_params must not have value at same time.
        assert not (bool(hps_params) and bool(gs_params))

        hps_mode = bool(hps_params)
        if hps_mode:
            return fixed_params, hps_mode, hps_params
        return fixed_params, hps_mode, gs_params

    def run_hps_experiment(self) -> str:
        """
        Args:
            None
        Returns:
            savename (str): _description_
        """
        # Prepare parameters.
        param_manager = ParameterManager(self.fixed_params)
        param_manager.add_param("seed", config["seed"]["hps"])
        param_manager.add_param("device", self.device)

        # Execute hyper parameter search.
        train_params = param_manager.get_parameter()
        csv_name = run_hps(
            train_params, self.save_loc, vars(self.search_params))

        # Copy hyperparameter result file.
        savename = os.path.join(
            self.save_loc, f"ResultTableHPS.csv")
        os.system(f"cp {csv_name} {savename}")


        return savename

    def run_evaluation_experiment(self, score_sheet: str):
        """

        Args:
            score_sheet (str): _description_
        """        
        param_manager = ParameterManager(self.fixed_params)
        param_manager.add_param("device", self.device)
        param_manager.update_by_search_result(
            score_sheet, vars(self.search_params), is_hps=True)

        # Evaluation run.
        seed = config["seed"]["hps"]
        param_manager.add_param("seed", seed)
        save_loc_train = os.path.join(
            self.save_loc, f"train_seed{seed:04d}")
        os.makedirs(save_loc_train, exist_ok=True)
        train_params = param_manager.get_parameter()

        # Run training and store result.
        result_dict, save_dir = run_train(train_params, save_loc_train)
        result_df = self._form_result_csv(
            result_dict["y_trues"][:, 0], 
            result_dict["y_preds"][:, 0]
        )

        savename = os.path.join(save_dir, "../..", "result.csv")
        result_df.to_csv(savename)

        return save_dir

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

    def main(self):
        """
        Args:
            None
        Returns:
            None
        """
        # Search.
        csv_path = self.run_hps_experiment()

        # Multi seed eval.
        result_file = self.run_evaluation_experiment(csv_path)
        

        # End

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        '--exp', 
        default=0
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

    executer = ExperimentManager(
        int(args.exp), 
        args.device,
        debug=args.debug
    )
    executer.main()
