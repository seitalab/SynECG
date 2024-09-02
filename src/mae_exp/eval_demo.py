import os
import pickle
from glob import glob
from typing import Optional
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

from run_eval import run_eval
from experiment import ExperimentManager
from codes.supports.monitor import sigmoid
from codes.supports.utils import get_timestamp, ResultManager, SlackReporter
from codes.supports.param_utils import ParameterManager

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)["experiment"]

dx_ds_dict = {
    "af": "AFIB",
    "iavb": "1AVB",
    "abqrs": "ABQRS",
    "asmi": "ASMI",
    "crbbb": "CRBBB",
    "imi": "IMI",
    "irbbb": "IRBBB",
    "isc": "ISC_",
    "lafb": "LAFB",
    "lvh": "LVH",
    "pac": "PAC",
    "pvc": "PVC",
    "std": "STD_",

    "norm": "NORM"
}

class DemographicsEvaluator(ExperimentManager):

    demos_target_file = "./resources/selected_for_demos.csv"

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
        self.target_model_root = self._get_trained_model_root(exp_id)
        if self.target_model_root is None:
            return

        exp_config_file = os.path.join(
            config["path"]["exp_root"],
            self.target_model_root[1:],
            f"exp{exp_id:04d}.yaml"
        )
        # Load parameters.
        fixed_params, self.hps_mode, search_params =\
            self._load_train_params(exp_config_file)
        fixed_params = self._merge_from_mae_setting(fixed_params)
        self.fixed_params = Namespace(**fixed_params)
        if not hasattr(self.fixed_params, "hps_result"):
            self.fixed_params.hps_result = None
        self.search_params = Namespace(**search_params)

        self._prepare_save_loc(exp_id)
        command = f"cp {exp_config_file} {self.save_loc}"
        os.system(command)
        self._select_device(device, use_cpu)

        self.debug = debug

    def _get_trained_model_root(self, exp_id: int):
        """
        Args:
            None
        Returns:
            str: _description_
        """
        # open `./resources/selected_for_demos.csv`
        df = pd.read_csv(self.demos_target_file, index_col=0)
        if exp_id not in df.exp_id.values:
            return None
        loc = df.loc[df.exp_id == exp_id, "hps_result"].values[0]
        return os.path.join(config["path"]["exp_root"], os.path.dirname(loc))

    def _get_trained_model_loc(self, seed: int):
        target_dir = os.path.join(
            config["path"]["exp_root"],
            self.target_model_root[1:], # remove the first "/"
            "multirun/train",
            f"seed{seed:04d}"
        )
        target_dir = glob(target_dir + "/??????-??????-*")[0]
        return target_dir

    def eval_multiseed_experiment(self, score_sheet: Optional[str], single_run: bool):
        """_summary_

        Args:
            score_sheet (str): _description_
        """        
        param_manager = ParameterManager(self.fixed_params)
        param_manager.add_param("device", self.device)
        if score_sheet is not None:
            param_manager.update_by_search_result(
                score_sheet, vars(self.search_params), is_hps=self.hps_mode)
        
        # Prepare result storer.
        savename = os.path.join(self.save_loc, "ResultTableMultiSeed.csv")
        result_manager = ResultManager(savename=savename)

        for s_idx, seed in enumerate(config["seed"]["multirun"]):
            
            param_manager.add_param("seed", seed)
            train_params = param_manager.get_parameter()
            trained_model_loc = self._get_trained_model_loc(seed)

            # Eval with PTBXL.
            save_loc_eval = os.path.join(
                self.save_loc, f"seed{seed:04d}")
            os.makedirs(save_loc_eval, exist_ok=True)

            if train_params.target_dx in config["dataset_dx"]["ptbxl"]:
                test_result_dict_ptbxl = run_eval(
                    eval_target=trained_model_loc, 
                    device=self.device,
                    dump_loc=save_loc_eval, 
                    eval_dataset="ptbxl",
                    multiseed_run=True
                )
                result_row = self._form_result_row(
                    seed, 
                    "ptbxl", 
                    test_result_dict_ptbxl["y_trues"][:, 0], 
                    test_result_dict_ptbxl["y_preds"][:, 0]
                )
                self._save_prediction_with_demos(
                    save_loc_eval, 
                    train_params.target_dx,
                    test_result_dict_ptbxl["y_trues"][:, 0], 
                    sigmoid(test_result_dict_ptbxl["y_preds"][:, 0])
                )
                result_manager.add_result(result_row)
            
            result_manager.save_result(is_temporal=True)
            if single_run:
                break

        result_manager.save_result()
        return result_manager.savename

    def _save_prediction_with_demos(self, save_loc: str, target_dx, y_true, y_pred):
        """
        Args:
            save_loc (str): _description_
            target_dx (str): _description_
            y_true (_type_): _description_
            y_pred (_type_): _description_
        """

        neg_data_demo = self._load_test_set_demos("norm")
        pos_data_demo = self._load_test_set_demos(target_dx)

        age = np.concatenate([neg_data_demo[:, 0], pos_data_demo[:, 0]])
        sex = np.concatenate([neg_data_demo[:, 1], pos_data_demo[:, 1]])

        assert (y_true == 0).sum() == len(neg_data_demo)

        save_loc = os.path.join(save_loc, "prediction.csv")
        pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "age": age,
            "sex": sex
        }).to_csv(save_loc)

    def _load_test_set_demos(self, target_dx: str):
        # Load demographics.
        target_data = os.path.join(
            config["path"]["data_root"],
            f"PTBXL-{dx_ds_dict[target_dx]}",
            "test_demo.pkl"
        )
        with open(target_data, "rb") as f:
            demos = pickle.load(f)
        return demos

    def main(self, single_run=False, hps_result=None):
        """
        Args:
            None
        Returns:
            None
        """

        # Overwrite.
        if self.fixed_params.hps_result is not None:
            assert hps_result is None
            hps_result = self.fixed_params.hps_result
        csv_path = hps_result
        
        # Multi seed eval.
        result_file = self.eval_multiseed_experiment(csv_path, single_run)
        
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
    # parser.add_argument(
    #     '--debug', 
    #     action="store_true"
    # )
    # parser.add_argument(
    #     '--multirun', 
    #     action="store_true"
    # )    
    # parser.add_argument(
    #     '--hps', 
    #     default=None
    # )    
    args = parser.parse_args()

    # print(args)
    # executer = DemographicsEvaluator(
    #     args.exp, 
    #     args.device,
    #     debug=False
    # )
    # executer.main(
    #     single_run=False,
    #     hps_result=None
    # )

    exp_ids = range(1, 5300)
    exp_ids = [5204, 5225, 5226, 5227, 5228, 5267, 5268]
    exp_ids = [5267]
    errors = []
    for exp_id in exp_ids:

        executer = DemographicsEvaluator(
            int(exp_id), 
            args.device,
            debug=False
        )
        if executer.target_model_root is None:
            continue

        try:
            executer.main(
                single_run=False,
                hps_result=None
            )
        except Exception as e:
            print(f"Error: {exp_id}")
            print(e)
            errors.append(exp_id)
    print("Done.")
    print(errors)
