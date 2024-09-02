from argparse import Namespace
from typing import Type

from torchvision import transforms
from torch.utils.data import DataLoader

from codes.data.dataset import ECGDataset
from codes.data.transform_funcs import (
    ToTensor, 
    RandomMask,
    RandomShift, 
    ScaleECG,
    AlignLength
)

def prepare_preprocess(
    params: Namespace, 
    is_train: bool,
    is_finetune: bool
) -> Type[transforms.Compose]:
    """
    Prepare and compose transform functions.
    Args:
        params (Namespace): 
        is_train (bool): 
    Returns:
        composed
    """
    transformations = []
    transformations.append(ScaleECG())
    transformations.append(
        AlignLength(int(params.max_duration * params.freq))
    )

    # Simple augmentations.
    if (is_train and is_finetune):
        transformations.append(RandomMask(params.mask_ratio))
        transformations.append(RandomShift(params.max_shift_ratio))

    # ToTensor and compose.
    transformations.append(ToTensor())
    composed = transforms.Compose(transformations)
    return composed

def prepare_dataloader(
    params: Namespace,
    datatype: str,
    is_train: bool,
    is_finetune: bool=False
) -> Type[DataLoader]:

    transformations = prepare_preprocess(params, is_train, is_finetune)
    
    data_lim = params.data_lim if is_train else params.val_lim 
    if params.dataset.find("//") != -1:
        target_dataset = None
        pos_dataset = params.dataset.split("//")[0]
        neg_dataset = params.dataset.split("//")[1]
    elif params.dataset is not None:
        target_dataset = params.dataset
        pos_dataset = None
        neg_dataset = None
    else:
        target_dataset = None
        pos_dataset = params.pos_dataset
        neg_dataset = params.neg_dataset

    dataset = ECGDataset(
        datatype, 
        params.seed, 
        pos_dataset,
        neg_dataset,
        target_dataset,
        data_lim,
        transformations
    )

    if not is_train:
        drop_last = False
    else:
        if params.data_lim is not None:
            data_lim = params.data_lim
        else:
            data_lim = 1e10
        if params.batch_size > data_lim:
            drop_last = False
        else:
            drop_last = True

    loader = DataLoader(
        dataset, 
        batch_size=params.batch_size, 
        shuffle=is_train, 
        drop_last=drop_last, 
        num_workers=params.n_workers
    )
    return loader
