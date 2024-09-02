import os
import ast
import pickle

import yaml
import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

class PTBXLPreparator:

    def __init__(self, target_dx: str, thres: float=100):

        self.target_dx = target_dx
        self.lead_idx = cfg["settings"]["ptbxl"]["lead_idx"]

        self.save_loc = os.path.join(
            cfg["settings"]["common"]["save_root"], 
            "PTBXL" + f"-{target_dx}"
        )
        os.makedirs(self.save_loc, exist_ok=True)

        self._prep_ecg(thres)

    def _prep_ecg(self, thres):
        """
        Args:

        Returns:

        """
        df = pd.read_csv(
            cfg["settings"]["ptbxl"]["src"] + "/../ptbxl_database.csv"
        )

        if self.target_dx == "ALL":
            df_target = df
        else:

            is_target = np.array([
                self.target_dx in ast.literal_eval(dx_dict).keys()
                for dx_dict in df.scp_codes.values
            ])
            df_dx = df[is_target]

            is_target = [
                ast.literal_eval(d)[self.target_dx] >= thres 
                for d in df_dx.scp_codes.values
            ]
            df_target = df_dx[is_target]

        ptbxl_ecgs = []
        demos = []
        for _, row in tqdm(df_target.iterrows(), total=df_target.shape[0]):
            target_id = row["ecg_id"]
            target_file = os.path.join(
                cfg["settings"]["ptbxl"]["src"], 
                f"{int(target_id/1000)*1000:05d}",
                f"{target_id:05d}_hr"
            )
            ecg = wfdb.rdrecord(target_file)
            ecg_ii = ecg.p_signal[:, ecg.sig_name.index("II")]

            if len(ecg_ii) != 5000:
                continue
            
            # error if `nan` exists.
            assert not np.isnan(ecg_ii).any()
            ptbxl_ecgs.append(ecg_ii)
            # extract demographic info from ecg.
            demos.append([row["age"], row["sex"]])
        self.demos = np.array(demos)
        self.ecgs = np.array(ptbxl_ecgs)
        assert len(self.ecgs) == len(self.demos)

    def _save_data(
        self, 
        data: np.ndarray, 
        demos: np.ndarray,
        datatype: str, 
        seed: int=None
    ):
        """
        Args:

        Returns:

        """
        if seed is not None:
            fname = f"{datatype}_seed{seed:04d}.pkl"
        else:
            fname = f"{datatype}.pkl"
        
        # save data.
        savename = os.path.join(
            self.save_loc,
            fname
        )
        
        with open(savename, "wb") as fp:
            pickle.dump(data, fp)

        # save demographic info.
        savename = os.path.join(
            self.save_loc,
            fname.replace(".pkl", "_demo.pkl")
        )
        with open(savename, "wb") as fp:
            pickle.dump(demos, fp)

    def make_dataset(self):
        """
        Args:

        Returns:

        """

        data_idxs = np.arange(len(self.ecgs))
        Xtr, Xte = train_test_split(
            data_idxs, 
            test_size=cfg["split"]["test"]["size"], 
            random_state=cfg["split"]["test"]["seed"]
        )
        self._save_data(self.ecgs[Xte], self.demos[Xte], "test")

        seeds = cfg["split"]["train_val"]["seeds"]
        for i, seed in enumerate(seeds):
            print(f"{i+1}/{len(seeds)}")
            Xtr_sp, Xv_sp = train_test_split(
                Xtr, 
                test_size=cfg["split"]["train_val"]["size"], 
                random_state=seed
            )
            self._save_data(self.ecgs[Xtr_sp], self.demos[Xtr_sp], "train", seed)
            self._save_data(self.ecgs[Xv_sp], self.demos[Xv_sp], "val", seed)
        print("Done")

if __name__ == "__main__":

    L_Thres = ["AFIB", "PAC", "STD_", "ABQRS"]

    # target_dx = "IRBBB"
    target_dxs = ["NORM", "AFIB", "CRBBB", "IRBBB", "PAC", "PVC"]
    target_dxs += ["STD_", "ASMI", "IMI"]
    target_dxs += ["LVH", "LAFB", "ISC_", "1AVB", "ABQRS"]
    for target_dx in target_dxs:
        print(target_dx)
        thres = 0 if target_dx in L_Thres else 100
        preparator = PTBXLPreparator(target_dx, thres)
        preparator.make_dataset()
    print("Done")
