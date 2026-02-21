import argparse
import os, random
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from moviad.models.padim.padim import Padim
from moviad.trainers.trainer_padim import TrainerPadim
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.evaluator import Evaluator, append_results
from moviad.utilities.configurations import TaskType, Split


BATCH_SIZE = 8
IMAGE_INPUT_SIZE = (224, 224)
OUTPUT_SIZE = (224, 224)


def main(args):
    mode = args.mode
    model_mode = args.model_mode
    backbone_id = args.backbone_id
    input_size = args.input_size
    output_size = args.output_size
    ad_layers = args.ad_layers
    batch_size = args.batch_size
    
    seeds = args.seeds
    device = args.device
    categories = args.categories
    save_path = args.save_path
    data_path = args.data_path
    results_dirpath = args.results_dirpath

    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        if "cuda" in device:
            torch.cuda.manual_seed_all(seed)

        for category_name in zip(categories):

            print("class name:", category_name)

            if args.train:
                print("---- PaDiM train ----")

                padim = Padim(
                    backbone_id,
                    category_name,
                    device=device,
                    model_mode=model_mode,
                    layers_idxs=ad_layers,
                )
                padim.to(device)

                train_dataset = MVTecDataset(
                    TaskType.SEGMENTATION,
                    data_path,
                    category_name,
                    Split.TRAIN,
                    img_size=input_size,
                )

                train_dataset.load_dataset()

                train_dataloader = DataLoader(
                    train_dataset, batch_size=batch_size, pin_memory=True
                )

                test_dataset = MVTecDataset(
                    TaskType.SEGMENTATION,
                    data_path,
                    category_name,
                    Split.TEST,
                    img_size=input_size,
                )

                test_dataset.load_dataset()

                test_dataloader = DataLoader(
                    test_dataset, batch_size=batch_size, pin_memory=True
                )
                
                trainer = TrainerPadim(
                    model=padim,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    device=device,
                    save_path=save_path,
                    model_mode=model_mode,
                )

                trainer.train()

                import pickle
                if save_path and category_name == "pill" and seed == 1:
                    print(f"--- Model saved in folder: {save_path} ---")
                    state_dict = padim.state_dict()
                    for hp in padim.HYPERPARAMS:
                        state_dict[hp] = getattr(padim, hp)
                    torch.save(state_dict, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL, _use_new_zipfile_serialization=True)

            if args.test:
                print("---- PaDiM test ----")

                # load the model if it was not trained in this run
                if not args.train:
                    padim = Padim(
                        backbone_id,
                        category_name,
                        device=device,
                        layers_idxs=ad_layers,
                    )
                    path = padim.get_model_savepath(save_path)
                    padim.load_state_dict(
                        torch.load(path, map_location=device, weights_only=False), strict=False
                    )
                    padim.to(device)
                    print(f"Loaded model from path: {path}")

                # Evaluator
                padim.eval()

                test_dataset = MVTecDataset(
                    TaskType.SEGMENTATION,
                    data_path,
                    category_name,
                    Split.TEST,
                    img_size=input_size,
                    gt_mask_size=output_size,
                )

                test_dataset.load_dataset()

                test_dataloader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=True
                )

                # evaluate the model
                evaluator = Evaluator(test_dataloader=test_dataloader, device=device)
                scores = evaluator.evaluate(padim)

                if results_dirpath is not None:
                    metrics_savefile = Path(
                        results_dirpath, f"metrics_{backbone_id}.csv"
                    )
                    # check if the metrics path exists
                    dirpath = os.path.dirname(metrics_savefile)
                    if not os.path.exists(dirpath):
                        os.makedirs(dirpath)

                    # save the scores
                    append_results(
                        metrics_savefile,
                        category_name,
                        seed,
                        *scores.values(),
                        "padim",  # ad_model
                        ad_layers,
                        backbone_id,
                        "IMAGENET1K_V2",  # NOTE: hardcoded, should be changed
                        None,  # bootstrap_layer
                        -1,  # epochs (not used)
                        args.img_input_size,
                        args.output_size,
                    )


if __name__ == "__main__":
    mvtec_categories = [
        "hazelnut", "bottle", "cable", "capsule", "carpet", "grid", "leather", "metal_nut", 
        "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
    ]
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Script execution mode: train or test")
    parser.add_argument("--model-mode", type=str, default="default", choices=["std", "diag", "sr"], help="Padim model mode. Standard: = 'std'; Diagonal = 'diag'; Super-Rank = 'sr'")
    parser.add_argument("--backbone-id", type=str, default=None, help="resnet18, wide_resnet50_2, mobilenet_v2, mcunet-in3")
    parser.add_argument("--ad_layers", type=int, nargs="+", required=True, help="list of layers idxs to use for feature extraction")
    parser.add_argument("--input_size", type=tuple[int], default=(224, 224), help="input image size, if None, default is used")
    parser.add_argument("--output_size", type=tuple[int], default=(224, 224), help="output image size, if None, default is used")
    parser.add_argument("--batch_size", type=int, default=32)
    
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--categories", type=str, nargs="+", default=mvtec_categories, help="Dataset category to test")
    parser.add_argument("--save_path", type=str, default=None, help="where to save the model checkpoint")
    parser.add_argument("--data_path", type=str, default="../../datasets/mvtec/")
    parser.add_argument("--results_dirpath", type=str, default=None)

    args = parser.parse_args()

    main(args)