import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.datasets.space_datasets import MarsDataset, LunarDataset


def idx_to_layer_name(backbone_model_name: str, idx: int) -> str | int:
    if backbone_model_name in ["wide_resnet50_2"]:
        return f"layer{idx}"
    elif backbone_model_name == "mobilenet_v2":
        return f"features.{idx}"
    else:
        return idx
    

def extract_raw_features(loader: DataLoader) -> list[torch.Tensor]:
    feats = []
    labels = []

    for imgs, lbls in loader:
        imgs_flat = imgs.view(imgs.size(0), -1)
        feats.append(imgs_flat.cpu())
        labels.append(lbls.cpu())

    return torch.cat(feats), torch.cat(labels)


def extract_backbone_features(feature_extractor: CustomFeatureExtractor, loader: DataLoader) -> list[torch.Tensor]:
    imgs_features = []
    imgs_labels = []

    feature_extractor.model.eval()

    with torch.no_grad():
        for imgs_batch, labels in loader:
            imgs_batch = imgs_batch.to("cuda")
            batch_features = feature_extractor(imgs_batch)[0]

            # Global average pooling
            batch_features = F.adaptive_avg_pool2d(batch_features, (1,1))
            batch_features = batch_features.view(batch_features.size(0), -1)

            imgs_features.append(batch_features.cpu())
            imgs_labels.append(labels)
            
    return torch.cat(imgs_features), torch.cat(imgs_labels)


def plot_tsne(X_tsne: np.ndarray, labels: np.ndarray, save_plot: str = None) -> None:
    dims = X_tsne.shape[1]

    class_names = np.unique(labels)

    if dims == 2:
        plt.figure(figsize=(8,6))
        
        for c in class_names:
            idx = labels == c
            plt.scatter(X_tsne[idx,0], X_tsne[idx,1], label=c, alpha=0.6)

        plt.legend()
        plt.title("PCA + t-SNE (2D)")

        if save_plot:
            plt.savefig(save_plot, dpi=300)

        plt.show()

    elif dims == 3:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="3d")

        for c in class_names:
            idx = labels == c
            ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], X_tsne[idx, 2], 
            label=c, alpha=0.6)

        ax.set_title("PCA + t-SNE (3D)")
        ax.legend()

        if save_plot:
            plt.savefig(save_plot, dpi=300, bbox_inches="tight")

        plt.show()

    else:
        print("Plot must be either 2D or 3D")

    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default=None, help="Path of the directory where the dataset is stored")    
    parser.add_argument("--backbone", type=str, default="mobilenet_v2", help="CNN model to use for feature extraction")
    parser.add_argument("--layer_idx", type=int, default=16, help="Backbone layer for feature extraction")
    parser.add_argument("--seed", type=int, default=1, help="Execution seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Runtime device")
    parser.add_argument("--view-dims", type=int, default=2, help="Number of dimensions in plot")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transform = None
    if not args.backbone:
        transform = transforms.Compose([transforms.ToTensor()])

    lunar_dataset = LunarDataset(root_dir="/home/fgenilotti/Downloads/vad-space/lunar/7041842", transform=transform)
    #mars_dataset = MarsDataset(root_dir="/home/fgenilotti/Downloads/vad-space/mars/3732485", transform=transform)
    
    loader = DataLoader(lunar_dataset, batch_size=32, shuffle=False)

    layer_idx = idx_to_layer_name(args.backbone, args.layer_idx)
    feature_extractor = CustomFeatureExtractor(
        model_name=args.backbone, 
        layers_idx=[layer_idx], 
        device=device, 
        frozen=True,
    )

    imgs_features, labels = None, None
    #args.backbone = None
    if args.backbone:
        imgs_features, labels = extract_backbone_features(
            feature_extractor=feature_extractor, 
            loader=loader,
        )
    else:
        imgs_features, labels = extract_raw_features(loader=loader)

    X = imgs_features.numpy()
    y = labels.numpy()

    n_pca = min(50, int(min(X.shape[0], X.shape[1]) / 2))
    pca = PCA(n_components=n_pca, random_state=args.seed)
    X_pca = pca.fit_transform(X)

    print("Explained variance:", pca.explained_variance_ratio_.sum())

    tsne = TSNE(
        n_components=args.view_dims,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=args.seed
    )

    X_tsne = tsne.fit_transform(X_pca)

    plot_tsne(X_tsne=X_tsne, labels=y)
    

if __name__ == "__main__":
    main()