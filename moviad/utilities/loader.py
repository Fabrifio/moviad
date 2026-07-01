import torch


def load_feature_extractor(config):
    import timm
    from moviad.utilities.helper import idx_to_layer_name
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor

    CNN_BACKBONES = ["mobilenet_v2", "wide_resnet50_2"]
    VIT_BACKBONES = ["deit_small_distilled_patch16_224", "deit_tiny_distilled_patch16_224"]

    backbone = config.backbone
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
        config.backbone,
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
    embed_dim = DEIT_CONFIGS[config.backbone]["embed_dim"]
    num_heads = DEIT_CONFIGS[config.backbone]["num_heads"] 

    # encoder vit
    encoder = vit_encoder.load(config.backbone)
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
        config.backbone,
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
        config.backbone,
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
        config.backbone, 
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