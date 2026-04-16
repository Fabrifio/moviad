import time
import random

from memory_profiler import memory_usage

import torch
from torch.utils.data import DataLoader


def idx_to_layer_name(backbone_model_name: str, idx: int) -> str | int:
    if backbone_model_name in ["wide_resnet50_2"]:
        return f"layer{idx}"
    elif backbone_model_name == "mobilenet_v2":
        return f"features.{idx}"
    else:
        return idx


def load_feature_extractor(config):
    import timm
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor

    CNN_BACKBONES = ["mobilenet_v2", "wide_resnet50_2"]
    VIT_BACKBONES = ["deit_small_distilled_patch16_224", "deit_tiny_distilled_patch16_224"]

    backbone = config.backbone_model_name
    device = config.device

    # CNN feature extractor
    if backbone in CNN_BACKBONES:
        if not hasattr(config, "ad_layers_idxs"):
            raise ValueError(f"{backbone} requires ad_layers_idxs")

        ad_layers = [
            idx_to_layer_name(backbone, idx)
            for idx in config.ad_layers_idxs
        ]

        feature_extractor = CustomFeatureExtractor(
            backbone,
            ad_layers,
            device
        )

    # ViT feature extractor
    elif backbone in VIT_BACKBONES:
        feature_extractor = timm.create_model(
            backbone,
            pretrained=True
        ).to(device)
        
        feature_extractor.eval()
        for p in feature_extractor.parameters():
            p.requires_grad = False

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return feature_extractor


def load_padim(config):
    from moviad.models.padim.padim import Padim

    # model init
    model = Padim(
        config.backbone_model_name,
        None,
        device=config.device,
        layers_idxs=config.ad_layers_idxs,
    )

    # load model
    state_dict = torch.load(
        config.save_path,
        map_location=config.device,
        weights_only=False
    )
    model.load_state_dict(state_dict, strict=False)

    model.to(config.device)
    model.eval()

    return model


def load_patchcore(config):
    from moviad.models.patchcore.patchcore import PatchCore

    # feature extractor
    feature_extractor = load_feature_extractor(config)

    # model init
    model = PatchCore(
        config.device,
        input_size=config.img_input_size,
        feature_extractor=feature_extractor,
        compression_method=None,
        apply_quantization=False
    )

    # load model
    model.load_model(config.save_path)

    model.to(config.device)
    model.eval()

    return model


def load_dinomaly(config):
    from moviad.Dinomaly.models.uad import ViTill
    from moviad.Dinomaly.models import vit_encoder
    from moviad.Dinomaly.models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
    from torch import nn
    from functools import partial

    # hard-coded config
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    DEIT_CONFIGS = {
        "deit_tiny_16": {"embed_dim": 192, "num_heads": 3},
        "deit_small_16": {"embed_dim": 384, "num_heads": 6},
        "deit_base_16": {"embed_dim": 768, "num_heads": 12},
    }
    embed_dim = DEIT_CONFIGS[config.backbone_model_name]["embed_dim"]
    num_heads = DEIT_CONFIGS[config.backbone_model_name]["num_heads"] 

    # encoder vit
    encoder = vit_encoder.load(config.backbone_model_name)
    encoder.to(config.device)

    # bottleneck
    bottleneck = nn.ModuleList([
        bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)
    ])

    # decoder vit
    decoder = nn.ModuleList([
        VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=LinearAttention2
        ) for _ in range(8)
    ])

    # model init
    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder
    )

    # load model
    state_dict = torch.load(config.save_path, map_location=config.device)
    model.load_state_dict(state_dict)

    model.to(config.device)
    model.eval()

    return model


def load_fastflow(config):
    from moviad.models.fastflow.fastflow import create_fastflow

    # model init
    model = create_fastflow(
        config.img_input_size,
        config.backbone_model_name,
        None,
        None,
        device=config.device
    )

    # load model
    state_dict = torch.load(config.save_path, map_location=config.device)
    model.load_state_dict(state_dict)

    model.to(config.device)
    model.eval()

    return model


def load_rd4ad(config):
    from moviad.models.rd4ad.rd4ad import RD4AD

    # model init
    model = RD4AD(
        config.backbone_model_name,
        config.device,
        input_size=config.img_input_size
    )

    # load model
    state_dict = torch.load(
        config.save_path,
        map_location=config.device,
        weights_only=False
    )
    model.load_state_dict(state_dict, strict=False)

    model.to(config.device)
    model.eval()

    return model


def load_cfa(config):
    from moviad.models.cfa.cfa import CFA

    feature_extractor = load_feature_extractor(config)

    # model init
    model = CFA(
        feature_extractor, 
        config.backbone_model_name, 
        config.device
    )

    # load model
    model.load_model(config.save_path)

    model.to(config.device)
    model.eval()

    return model


def load_stfpm(config):
    from moviad.models.stfpm.stfpm import STFPM

    teacher_feature_extractor = load_feature_extractor(config)
    student_feature_extractor = load_feature_extractor(config)

    # model init
    model = STFPM(
        teacher_feature_extractor,
        student_feature_extractor
    )

    # load model
    state_dict = torch.load(
        config.save_path,
        map_location=config.device,
        weights_only=False
    )
    model.load_state_dict(state_dict, strict=False)

    model.to(config.device)
    model.eval()

    return model


def load_ssnet(config):
    from moviad.models.supersimplenet.supersimplenet import SuperSimpleNet

    feature_extractor = load_feature_extractor(config)

    # model init
    model = SuperSimpleNet(
        feature_extractor
    )

    # load model
    state_dict = torch.load(
        config.save_path,
        map_location=config.device,
        weights_only=False
    )
    model.load_state_dict(state_dict, strict=False)

    model.to(config.device)
    model.eval()

    return model


def load_model(name: str, args):
    name = name.lower()

    if name == "padim":
        return load_padim(args)
    elif name == "patchcore":
        return load_patchcore(args)
    elif name == "fastflow":
        return load_fastflow(args)
    elif name == "cfa":
        return load_cfa(args)
    elif name == "dinomaly":
        return load_dinomaly(args)
    elif name == "stfpm":
        return load_stfpm(args)
    elif name == "rd4ad":
        return load_rd4ad(args)
    else:
        raise ValueError(f"Unknown model: {name}")


def run_forward(model, images):
    with torch.no_grad():
        return model(images)


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_gpu_stats(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_gpu_peak(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return -1


def dynamic_profile(model, dataloader, device, num_batches=10):
    """
    Profile VAD model dynamically over the dataloader batches. 
    It measures latency per sample, GPU max memory peak and 
    deltas of the CPU memory. 
    
    Args:
        model: loaded VAD model
        dataloader: dataloader for the profiling
        device: model device
        num_batches: number of batches to profile
    """
    
    latencies = []
    cpu_mem_peaks = []
    gpu_mem_peak = -1

    # warm-up GPU and CPU
    images = next(iter(dataloader))[0].to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(images)

    # loop over batches
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        images = batch[0].to(device)

        #reset_gpu_stats(device)

        # latency
        with torch.no_grad():
            #sync_if_needed(device)
            t0 = time.perf_counter()

            _ = model(images)

            #sync_if_needed(device)
            t1 = time.perf_counter()

        latencies.append((t1 - t0) / images.size(0))

        # GPU memory
        gpu_mem_peak = 0#max(gpu_mem_peak, get_gpu_peak(device))

        # CPU memory
        mem_trace = memory_usage(
            (run_forward, (model, images)),
            interval=0.001,
            max_iterations=1,
            retval=False
        )

        cpu_mem_peaks.append(max(mem_trace) - min(mem_trace))

    # report results
    print("\n--- Profiling Results ---")
    print(f"Average latency per sample: {sum(latencies)/len(latencies)*1000:.2f} ms")
    print(f"Max CPU memory delta per batch: {max(cpu_mem_peaks):.2f} MB")
    print(f"Max GPU memory usage: {gpu_mem_peak/1024**2:.2f} MB")

    return {
        "latencies_ms": [l*1000 for l in latencies],
        "cpu_mem_peaks_mb": cpu_mem_peaks,
        "gpu_mem_peak_mb": gpu_mem_peak / 1024**2,
    }


def main(args):
    device = torch.device(args.device)
    seeds = args.seeds if isinstance(args.seeds, (list, tuple)) else [args.seeds]

    for seed in seeds:
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if "cuda" in device.type:
            torch.cuda.manual_seed_all(seed)

        # load model
        model = load_model(args.model, args)
        model.to(device)

        print(f"[INFO] Loaded {args.model} | seed={seed}")

        from torchvision import transforms
        from moviad.datasets.ad_datasets import AnoVoxDataset

        # define torchvision transformations
        transform = transforms.Compose([
            transforms.Resize(
                (224,224),
                antialias=True,
            ),
            transforms.ToTensor()
        ])
        sem_transform = transforms.Compose([
            transforms.Resize(
                (224,224),
                antialias=True,
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ]) 

        test_dataset = AnoVoxDataset(
            root_dir=args.dataset_path, 
            mode="test", 
            transform=transform, 
            sem_transform=sem_transform
        )

        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            pin_memory=True
        )

        print(f"[INFO] Loaded dataset \n[INFO] Starting profiling")

        # dynamic profile
        dynamic_profile(
            model=model, 
            dataloader=test_dataloader, 
            device=device, 
            num_batches=getattr(args, "num_batches", 1000)
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        help="Supported models: cfa; dinomaly; fastflow; padim; patchcore; rd4ad; ssnet; stfpm"
    )
    parser.add_argument(
        "--backbone_model_name",
        type=str,
        help="Supported backbones: mobilenet_v2; wide_resnet50_2; deit_small_distilled_patch16_224; deit_tiny_distilled_patch16_224",
    )
    parser.add_argument(
        "--img_input_size",
        type=int,
        nargs=2,
        default=(224, 224),
        help="Input image size: if None, default is used",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32
    )
    parser.add_argument(
        "--num_batches", 
        type=int, 
        default=1,
        help="Number of batches used to profile the model"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None, 
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default=None
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="cpu, cuda"
    )
    parser.add_argument(
        "--seeds", 
        type=int, 
        nargs="+", 
        default=[1, 2, 7]
    )
    parser.add_argument(
        "--ad_layers_idxs",
        type=int,
        nargs="+",
        required=True,
        help="List of layers idxs to use for CNN feature extraction",
    )

    args = parser.parse_args()

    print("---- VAD Model Profiler ----")
    main(args)