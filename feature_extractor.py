"""Extracts and caches image features using a pretrained CNN (transfer learning)."""
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import config
from src.dataset import ImageOnlyDataset


class CNNFeatureExtractor(nn.Module):
    """Wraps a pretrained torchvision CNN, chopped off before its classification head.

    Output is a single pooled feature vector per image (shape: FEATURE_DIM,),
    which keeps the decoder simple. Swap `_build_backbone` for a spatial-feature
    version (e.g. keep the last conv map) if you want a spatial attention decoder.
    """

    def __init__(self, backbone_name: str = config.CNN_BACKBONE):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = self._build_backbone(backbone_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False  # transfer learning: extractor is frozen

    @staticmethod
    def _build_backbone(name):
        if name == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            modules = list(net.children())[:-1]  # drop the final fc layer
            return nn.Sequential(*modules, nn.Flatten())
        if name == "inceptionv3":
            net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
            net.fc = nn.Identity()
            net.aux_logits = False
            return net
        if name == "efficientnet_b0":
            net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            net.classifier = nn.Identity()
            return net
        raise ValueError(f"Unknown backbone: {name}")

    @staticmethod
    def preprocess_transform():
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    @torch.no_grad()
    def extract_and_cache(self, image_ids, images_dir: str = config.IMAGES_DIR,
                           out_dir: str = config.FEATURES_DIR, device=config.DEVICE, batch_size=32):
        """Runs the CNN once per image and saves each feature vector as a .npy file,
        so training epochs never re-run the (frozen) CNN forward pass."""
        os.makedirs(out_dir, exist_ok=True)
        remaining = [i for i in image_ids if not os.path.exists(os.path.join(out_dir, i + ".npy"))]
        if not remaining:
            print("All features already cached.")
            return

        dataset = ImageOnlyDataset(remaining, images_dir, self.preprocess_transform())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)

        self.backbone.to(device)
        for image_ids_batch, images in tqdm(loader, desc=f"Extracting features ({self.backbone_name})"):
            images = images.to(device)
            features = self.forward(images).cpu().numpy()
            for img_id, feat in zip(image_ids_batch, features):
                np.save(os.path.join(out_dir, img_id + ".npy"), feat.astype("float32"))


if __name__ == "__main__":
    from src.dataset import Flickr8kCaptions

    data = Flickr8kCaptions()
    all_ids = data.df["image"].unique().tolist()
    extractor = CNNFeatureExtractor()
    extractor.extract_and_cache(all_ids)
