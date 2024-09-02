from datetime import datetime
from typing import Dict, List

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from slack_sdk import WebClient

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)
slack_config = config["slack"]

def get_timestamp() -> str:
    """
    Get timestamp in `yymmdd-hhmmss` format.
    Args:
        None
    Returns:
        timestamp (str): Time stamp in string.
    """
    timestamp = datetime.now()
    timestamp = timestamp.strftime('%Y%m%d-%H%M%S')[2:]
    return timestamp

def calc_class_weight(
    labels: List
) -> np.ndarray:
    """
    Calculate class weight for multilabel task.
    Args:
        labels (np.ndarray): Label data array of shape [num_sample, num_classes]
    Returns:
        class_weight (np.ndarray): Array of shape [num_classes].
    """
    labels = np.array(labels)[:, np.newaxis]
    num_samples = labels.shape[0]

    positive_per_class = labels.sum(axis=0)
    negative_per_class = num_samples - positive_per_class

    class_weight = negative_per_class / positive_per_class

    return class_weight

def aggregator(model: nn.Module, X: torch.Tensor):
    """
    ```
    These predictions are then aggregated using the element-wise maximum
    (or mean in case of age and gender prediction)
    ```
    From paper's appendix I => We use maximum.
    Args:
        model (nn.Module):
        X (torch.Tensor): Tensor of shape (batch_size, num_split, sequence_length).
    Returns:
        aggregated_preds (torch.Tensor): Tensor of shape (batch_size, num_classes).
    """
    aggregated_preds = []
    for i in range(X.size(0)):
        if X[i].dim() == 3:
            data = torch.permute(X[i], (1, 0, 2))
        else:
            data = X[i]
        y_preds = model(data) # data: (num_split, sequence_length)
        _aggregated_preds, _ = torch.max(y_preds, axis=0)
        aggregated_preds.append(_aggregated_preds)
    aggregated_preds = torch.stack(aggregated_preds)
    return aggregated_preds

class SlackReporter:

    def __init__(self):
        """
        Args:
            None
        Returns:
            None
        """        
        self.client = WebClient(
            token=slack_config["token"]
        )
        self.channel_id = slack_config["channel_id"]

    def report(
        self, 
        message: str,
        parent_message: str, 
    ) -> None:
        """
        Args:
            message (str): 
            parent_message (str): 
        Returns:
            None
        """
        history = self.client.conversations_history(
            channel=self.channel_id
        )
        posts = history["messages"][:slack_config["max_past"]]

        for post in posts:
            if post["text"] != parent_message:
                continue
            self.client.chat_postMessage(
                channel=self.channel_id,
                thread_ts=post["ts"],
                text=message
            )
            break
    
    def post(self, message: str):
        """
        Args:
            message (str): 
        Returns:
            None
        """
        self.client.chat_postMessage(
            text=message, 
            channel=self.channel_id,
        )

class ResultManager:

    def __init__(self, savename: str):
        self.savename = savename
        self.results = []

    def add_result(self, row: Dict):
        """
        Add one row to results.
        
        Args:
            row (List): _description_
        Returns: 
            None
        """
        self.results.append(row)

    def get_result_df(self) -> pd.DataFrame:
        """
        Args:
            None
        Returns:
            df_result: 
        """
        df_result = pd.DataFrame(self.results)
        return df_result
    
    def save_result(self, is_temporal: bool=False):
        """

        Args:
            is_temporal (bool, optional): _description_. Defaults to False.
        """
        df_result = pd.DataFrame(self.results)
        
        savename = self.savename
        if is_temporal:
            savename = savename.replace(".csv", "_tmp.csv")
        df_result.to_csv(savename)
