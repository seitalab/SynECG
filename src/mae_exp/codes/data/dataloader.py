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
    AlignLength,
    Subsample,
    SubsampleEval
)

def prepare_preprocess(
    params: Namespace, 
    is_train: bool,
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
    if params.neg_dataset.startswith("CPSC"):
        if is_train:
            transformations.append(
                Subsample(int(params.max_duration * params.freq))
            )
        else:
            transformations.append(
                SubsampleEval(int(params.max_duration * params.freq))
            )
    else:
        transformations.append(
            AlignLength(int(params.max_duration * params.freq))
        )

    # Simple augmentations.
    if is_train:
        transformations.append(RandomMask(params.mask_ratio))
        transformations.append(RandomShift(params.max_shift_ratio))

    # ToTensor and compose.
    transformations.append(ToTensor(params.modelname))
    composed = transforms.Compose(transformations)
    return composed

def prepare_dataloader(
    params: Namespace,
    datatype: str,
    is_train: bool,
) -> Type[DataLoader]:

    transformations = prepare_preprocess(params, is_train)
    
    data_lim = params.data_lim if is_train else params.val_lim 
    pos_dataset = params.pos_dataset
    neg_dataset = params.neg_dataset

    dataset = ECGDataset(
        datatype, 
        params.seed, 
        pos_dataset,
        neg_dataset,
        data_lim,
        transformations
    )

    if not is_train:
        drop_last = False
    else:
        if params.data_lim is not None:
            data_lim = params.data_lim
            if type(data_lim) == str:
                assert data_lim[-1] in ["n", "p"]
                data_lim = int(data_lim[:-1])
        else:
            data_lim = 1e10
        if params.batch_size > data_lim:
            drop_last = False
        else:
            drop_last = True
    if params.neg_dataset.startswith("CPSC"):
        if not is_train:
            batch_size = 1
        else:
            batch_size = params.batch_size
    else:
        batch_size = params.batch_size

    loader = DataLoader(
        dataset, 
        # batch_size=params.batch_size, 
        batch_size=batch_size,
        shuffle=is_train, 
        drop_last=drop_last, 
        num_workers=params.n_workers
    )
    return loader
