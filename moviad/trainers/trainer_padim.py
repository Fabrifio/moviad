import os
from tqdm import tqdm
import torch

from moviad.models.padim.padim import Padim
from moviad.trainers.trainer import Trainer, TrainerResult


class TrainerPadim(Trainer):

    def __init__(
        self, 
        model: Padim,  
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        device,
        model_mode="std",
        save_path=None,
        logger=None,
    ):
        """
        Args:
            device: one of the following strings: 'cpu', 'cuda', 'cuda:0', ...
        """
        super().__init__(
            model, 
            train_dataloader, 
            test_dataloader, 
            device, 
            logger,
            save_path,
        )
        self.model_mode = model_mode

    def train(self):
        print(f"Train Padim. Backbone: {self.model.backbone_model_name}")

        self.model.train()

        if self.logger is not None:
            self.logger.watch(self.model)

        # 1. get the feature maps from the backbone
        layer_outputs: dict[str, list[torch.Tensor]] = {
            layer: [] for layer in self.model.layers_idxs
        }
        for x in tqdm(
            self.train_dataloader, "| feature extraction | train | %s |" 
        ):
            outputs = self.model(x.to(self.device))
            assert isinstance(outputs, dict)
            for layer, output in outputs.items():
                layer_outputs[layer].extend(output)

        # 2. use the feature maps to get the embeddings
        embedding_vectors = self.model.raw_feature_maps_to_embeddings(layer_outputs)
        
        # 3. fit the multivariate Gaussian distribution
        if self.model_mode == "std":
            self.model.fit_multivariate_gaussian(embedding_vectors, update_params=True, logger=self.logger)
        elif self.model_mode == "diag":
            self.model.fit_multivariate_diagonal_gaussian(embedding_vectors, update_params=True, logger=self.logger)
        elif self.model_mode == "sr":
            self.model.fit_multivariate_diagonal_gaussian(embedding_vectors, update_params=True, logger=self.logger)
            self.model.fit_pca_sr(embedding_vectors, update_params=True)

        metrics = {
            "img_roc_auc": 0.1,
            "pxl_roc_auc": 0.1,
            "img_f1": 0.1,
            "pxl_f1": 0.1,
            "img_pr_auc": 0.1,
            "pxl_pr_auc": 0.1,
            "pxl_au_pro": 0.1
        } #self.evaluator.evaluate(self.model)

        #if self.logger is not None:
        #    self.logger.log(
        #        metrics
        #    )
#
        #print("End training performances:")
        #self.print_metrics(metrics)

        return TrainerResult(**metrics)