import os
import socket
from argparse import Namespace

import yaml

from run_train import run_train
from experiment import ExperimentManager
from codes.supports.utils import get_timestamp

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

class PretrainManager(ExperimentManager):

    def __init__(
        self, 
        pretrain_id: int, 
        device: str, 
        use_cpu: bool=False,
        debug: bool=False
    ):
        pt_config_file = os.path.join(
            config["experiment"]["path"]["pretrain_yaml_loc"],
            f"p{pretrain_id//100:02d}s",
            f"pt{pretrain_id:04d}.yaml"
        )

        fixed_params, _, _ =\
            self._load_train_params(pt_config_file)
        fixed_params = self._update_fixed_params(fixed_params)
        self.fixed_params = Namespace(**fixed_params)

        self.fixed_params.seed = config["experiment"]["seed"]["pretrain"]
        self.fixed_params.host = socket.gethostname()
        self.fixed_params.device = device
        self.fixed_params.target_dx = "<none; pretrain>"
        self.fixed_params.pos_dataset = "<none; pretrain>"
        self.fixed_params.neg_dataset = "<none; pretrain>"

        self._prepare_save_loc(pretrain_id)
        self._save_config(pt_config_file)
        self._select_device(device, use_cpu)

        # For debugging.
        if debug:
            self.fixed_params.epochs = 2
            self.fixed_params.eval_every = 1
        self.debug = debug

    def _update_fixed_params(self, fixed_params):

        key = fixed_params["pretrain_key"]
        fixed_params.update(config["pretrain_params"]["mae"][key])

        # str -> float
        fixed_params["total_samples"] = self._str_to_number(
            fixed_params["total_samples"])
        fixed_params["eval_every"] = self._str_to_number(
            fixed_params["eval_every"])
        fixed_params["learning_rate"] = self._str_to_number(
            fixed_params["learning_rate"], to_int=False)
        if "save_model_every" in fixed_params:
            fixed_params["save_model_every"] = self._str_to_number(
                fixed_params["save_model_every"])

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
            config["path"]["processed_data"],
            config["experiment"]["path"]["save_root"],
            f"mae-pt{pretrain_id:04d}"[:-2]+"s",
            f"pt{pretrain_id:04d}",
            get_timestamp()
        )
        os.makedirs(self.save_loc, exist_ok=True)

    def main(self, single_run=False):
        """
        Args:
            None
        Returns:
            None
        """
        _, save_dir = run_train(self.fixed_params, self.save_loc)
        print(save_dir)

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        '--pt', 
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

    executer = PretrainManager(
        int(args.pt), 
        args.device,
        debug=args.debug
    )
    executer.main()
