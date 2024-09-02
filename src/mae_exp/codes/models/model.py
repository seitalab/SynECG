import sys
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")
from codes.models.mae import mae_vit_base
from baselines.codes.architectures.transformer import LinearEmbed, Transformer

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
    is_mae = False
    
    if params.modelname == "mae_base":
        model_backbone = mae_vit_base(params)
        foot = None
        is_mae = True
    
    elif params.modelname == "transformer":
        foot = LinearEmbed(params)
        model_backbone = Transformer(params)

    elif params.modelname == "luna":
        from baselines.codes.architectures.luna import LunaTransformer
        foot = LinearEmbed(params)
        model_backbone = LunaTransformer(params)

    elif params.modelname == "resnet18":
        from baselines.codes.architectures.resnet import ResNet18
        foot = None
        model_backbone = ResNet18(params)    

    elif params.modelname == "emblstm":
        from baselines.codes.architectures.bi_lstm import VarDepthLSTM
        foot = LinearEmbed(params, add_cls_token=False)
        model_backbone = VarDepthLSTM(params,  params.emb_dim)

    elif params.modelname == "embgru":
        from baselines.codes.architectures.bi_gru import VarDepthGRU
        foot = LinearEmbed(params, add_cls_token=False)
        model_backbone = VarDepthGRU(params, params.emb_dim)

    elif params.modelname == "mega":
        from baselines.codes.architectures.mega import Mega
        foot = LinearEmbed(params)
        model_backbone = Mega(params)

    elif params.modelname == "s4":
        from baselines.codes.architectures.s4 import S4
        foot = LinearEmbed(params)
        model_backbone = S4(params)

    elif params.modelname == "resnet34":
        from baselines.codes.architectures.resnet import ResNet34
        foot = None
        model_backbone = ResNet34(params)   

    elif params.modelname == "resnet50":
        from baselines.codes.architectures.resnet import ResNet50
        foot = None
        model_backbone = ResNet50(params)   

    elif params.modelname == "effnetb0":
        sys.path.append("../baselines")
        from baselines.codes.architectures.efficient_net import effnet1d_b0
        foot = None
        seqlen = int(
            (params.max_duration * params.freq / params.downsample)
        )
        effnet_params = {
            "num_lead": params.num_lead,
            "sequence_length": seqlen,
            "backbone_out_dim": params.backbone_out_dim
        }        
        model_backbone = effnet1d_b0(**effnet_params)

    else:
        raise NotImplementedError(f"{params.modelname} is not implemented.")

    if not hasattr(params, "emb_dim"):
        emb_dim = None
    else:
        emb_dim = params.emb_dim

    head = HeadModule(params.backbone_out_dim)
    model = Predictor(
        model_backbone, 
        head, 
        emb_dim, 
        params.backbone_out_dim, 
        foot,
        is_mae=is_mae
    )
    return model

class Predictor(nn.Module):

    def __init__(
        self, 
        backbone: nn.Module, 
        head: nn.Module,
        emb_dim: int,
        backbone_out_dim: int,
        foot: nn.Module=None,
        is_mae: bool=True
    ) -> None:
        super(Predictor, self).__init__()

        self.backbone = backbone
        self.head = head
        self.foot = foot

        if emb_dim is not None:
            self.fc = nn.Linear(emb_dim, backbone_out_dim)
        self.is_mae = is_mae

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size 
                (batch_size, num_lead, seq_len).
        Returns:
            h (torch.Tensor): Tensor of size (batch_size, num_classes)
        """
        if self.is_mae:
            h, _, _ = self.backbone.forward_encoder(x, mask_ratio=0) # (bs, num_chunks, emb_dim)

            # Calc average of tokens.
            h = torch.mean(h, dim=1)
            h = self.fc(h)
        else:
            if self.foot is not None:
                x = self.foot(x)
            h = self.backbone(x)
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