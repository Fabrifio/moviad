import random

import torch
import numpy as np


def set_seed(seed: int):
    """
    Set the random, torch and numpy seed for reproducibility.
    Args:
        seed (int): The seed to set.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def idx_to_layer_name(backbone: str, idx: int) -> str | int:
    """
    Convert a layer index to a layer id based on the backbone model. 
    Supported backbones are: wide_resnet50_2 and mobilenet_v2. 
    If the backbone is not supported, the index is returned as is.
    Args:
        backbone (str): The name of the backbone model.
        idx (int): The index of the layer.
    Returns:
        str | int: The layer id based on the backbone model.
    """
    if backbone in ["wide_resnet50_2"]:
        return f"layer{idx}"
    elif backbone == "mobilenet_v2":
        return f"features.{idx}"
    else:
        return idx
    

def get_ad_layers_ids(backbone: str, ad_layers: list) -> list:
    """
    Convert a list of layer indices to a list of layer ids based on the backbone model.
    Args:
        backbone (str): The name of the backbone model.
        ad_layers (list): A list of layer indices.
    Returns:
        list: A list of layer ids based on the backbone model.
    """
    return [
        idx_to_layer_name(backbone=backbone, idx=idx)
        for idx in ad_layers
    ]