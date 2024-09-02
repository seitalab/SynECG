import os
import pickle
from glob import glob
from typing import Optional, List

import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

# root = "/home/nonaka/git/ecg_pj/SynthesizedECG/data"
cfg_file = "../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

class ECGDataset(Dataset):

    def __init__(
        self, 
        datatype: str,
        seed: int,
        pos: str,
        neg: str,
        dataset: str,
        data_lim: Optional[int]=None, 
        transform: Optional[List]=None
    ) -> None:
        """
        Args:
            root (str): Path to dataset directory.
            datatype (str): Dataset type to load (train, valid, test)
            seed (int): 
            pos (str): Positive label dataset.
            neg (str): Negative label dataset.
            data_lim (int): Total number of samples (load data_lim/2 samples from pos/neg dataset each).
                If data_lim = 1000: pos = 500, neg = 500.
                In case, number of samples in pos/neg dataset is below data_lim / 2, 
                total number of samples used will be less than data_lim.
            transform (List): List of transformations to be applied.
        """
        assert(datatype in ["train", "val", "test"])
        
        self.data_loc = cfg["experiment"]["path"]["data_root"]

        # Limit total number of dataset    
        self.data, self.label = [], []
        if dataset is not None:
            self.data_lim = data_lim #if data_lim is not None else data_lim
            data = self._load_data(datatype, seed, dataset)
            for d in data:
                self.data.append(d)
                self.label.append(0)
        else:
            self.data_lim = data_lim // 2 if data_lim is not None else data_lim
            data = self._load_data(datatype, seed, neg)
            for d in data:
                self.data.append(d)
                self.label.append(0)
            data = self._load_data(datatype, seed, pos)
            for d in data:
                self.data.append(d)
                self.label.append(1)

        print(f"Loaded {datatype} set: {len(self.data)} samples.")

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        if self.transform:
            sample = {"data": data}
            sample = self.transform(sample)
            data = sample["data"]

        return data, torch.Tensor([self.label[index]])

    def _load_data(self, datatype: str, seed: int, dirname: str) -> np.ndarray:
        """
        Load file of target datatype.
        Args:
            datatype (str)
        Returns:
            X (np.ndarray): Array of shape (num_samples, 12, sequence_length).
        """
        if datatype == "test":
            filename = "test.pkl"
        else:
            filename = f"{datatype}_seed{seed:04d}.pkl"

        if dirname.startswith("gen-"):
            target_info = dirname.split("-")[1]
            dgm_used = target_info.split("/")[0]
            data_ver = target_info.split("/")[1]
            data_loc = os.path.join(
                cfg["generatives"]["data"][dgm_used][data_ver],
                datatype,
                "samples"
            )
            targets = sorted(glob(data_loc + "/idx*.pkl"))
            ecg_data = []
            print("Loading generated data...")
            for target in tqdm(targets):
                with open(target, "rb") as fp:
                    ecg_data.append(pickle.load(fp))
            ecg_data = np.concatenate(ecg_data, axis=0)
            ecg_data = ecg_data[:, 0] # n_data, 1, seqlen => n_data, seqlen

        else:
            target = os.path.join(
                self.data_loc,
                dirname, 
                filename
            )
            with open(target, "rb") as fp:
                ecg_data = pickle.load(fp)

        # Randomly shuffle data if real data with data lim.
        # if not dirname.startswith("syn_"):
        #     if self.data_lim is not None:
        #         np.random.seed(seed)
                # np.random.shuffle(ecg_data)
                

        if self.data_lim is not None:
            print(f"WARNING: LIMITING NUMBER OF SAMPLES: {len(ecg_data)} -> {self.data_lim}")
            if self.data_lim < 500000:
                # This is probably faster for small datasize 
                # but process is killed if datasize is large.
                np.random.seed(seed)
                idxs = np.random.choice(
                    len(ecg_data), 
                    size=self.data_lim, 
                    replace=False
                )
                ecg_data_lim = np.array(ecg_data[idxs])
                del ecg_data
                return ecg_data_lim
            else:
                np.random.seed(seed)
                np.random.shuffle(ecg_data)
                return ecg_data[:self.data_lim]
        return ecg_data
    
if __name__ == "__main__":
    pass