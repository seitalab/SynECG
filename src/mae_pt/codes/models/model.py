from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from codes.models.mae import mae_vit_base

def prepare_mae_model(params: Namespace) -> nn.Module:
    """
    Args:
        params (Namespace):
    Returns:
        predictor (nn.Module):
    """
    if params.modelname == "mae_base":
        model = mae_vit_base(params)
    else:
        raise NotImplementedError(
            f"{params.modelname} not available")
    return model

def prepare_clf_model(params: Namespace) -> nn.Module:
    """
    Args:
        params (Namespace):
    Returns:
        predictor (nn.Module):
    """
    if params.modelname == "mae_base":
        model_backbone = mae_vit_base(params)
    
    if params.clf_mode == "logistic_regression":
        model = Classifier(model_backbone, params.emb_dim)
    else:
        head = HeadModule(params.clf_fc_dim)
        model = Predictor(
            model_backbone, 
            head, 
            params.emb_dim, 
            params.clf_fc_dim, 
            params.select_type
        )
    return model

class Classifier(nn.Module):

    def __init__(self, mae, emb_dim):
        super(Classifier, self).__init__()

        self.mae = mae
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, x):

        h, _, _ = self.mae.forward_encoder(x, mask_ratio=0) # (bs, num_chunks, emb_dim)
        h = self.fc(h[:, 0]) # use cls_token.
        return h

class Predictor(nn.Module):

    def __init__(
        self, 
        mae: nn.Module, 
        head: nn.Module,
        emb_dim: int,
        backbone_out_dim: int,
        select_type: str="cls_token"
    ) -> None:
        super(Predictor, self).__init__()

        self.mae = mae
        self.head = head
        self.fc = nn.Linear(emb_dim, backbone_out_dim)
        self.select_type = select_type

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size 
                (batch_size, num_lead, seq_len).
        Returns:
            h (torch.Tensor): Tensor of size (batch_size, num_classes)
        """
        h, _, _ = self.mae.forward_encoder(x, mask_ratio=0) # (bs, num_chunks, emb_dim)

        if self.select_type == "cls_token":
            h = h[:, 0]
        elif self.select_type == "mean":
            h = torch.mean(h, dim=1)
        else:
            raise NotImplementedError
        h = self.fc(h)
        h = self.head(h)
        return h

class HeadModule(nn.Module):

    def __init__(self, in_dim: int):
        super(HeadModule, self).__init__()

        self.fc1 = nn.Linear(in_dim, 32)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size (num_batch, in_dim).
        Returns:
            feat (torch.Tensor): Tensor of size (num_batch, 1).
        """
        feat = F.relu(self.fc1(x))
        feat = self.drop1(feat)
        feat = self.fc2(feat)
        return feat