import os
import ast

import yaml
import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from ptbxl import PTBXLPreparator

cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

class PTBXLExtraLeadPreparator(PTBXLPreparator):

    def __init__(
        self, 
        target_dx: str, 
        target_lead: str,
        thres: float=100
    ):

        self.target_dx = target_dx
        self.target_lead = target_lead

        self.save_loc = os.path.join(
            cfg["settings"]["common"]["save_root"], 
            "PTBXL" + f"-{target_dx}-LEAD_{target_lead}"
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
        for target_id in tqdm(df_target.ecg_id.values):
            target_file = os.path.join(
                cfg["settings"]["ptbxl"]["src"], 
                f"{int(target_id/1000)*1000:05d}",
                f"{target_id:05d}_hr"
            )
            ecg = wfdb.rdrecord(target_file)
            ecg_lead = ecg.p_signal[:, ecg.sig_name.index(self.target_lead)]

            if len(ecg_lead) != 5000:
                continue
            
            # error if `nan` exists.
            assert not np.isnan(ecg_lead).any()

            ptbxl_ecgs.append(ecg_lead)
        self.ecgs = np.array(ptbxl_ecgs)

if __name__ == "__main__":

    L_Thres = ["AFIB", "PAC", "STD_", "ABQRS"]

    target_dxs = [
        "NORM", "AFIB", "CRBBB", "IRBBB", "PAC", "PVC", "STD_",
        "ASMI", "IMI", "LVH", "LAFB", "ISC_", "1AVB", "ABQRS"
    ]
    leads = [
        "I", "II", "III", "AVR", "AVL", "AVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]
    for target_dx in target_dxs:
        for lead in leads:
            print(target_dx, lead)
            thres = 0 if target_dx in L_Thres else 100
            preparator = PTBXLExtraLeadPreparator(target_dx, lead, thres)
            preparator.make_dataset()
    print("Done")
