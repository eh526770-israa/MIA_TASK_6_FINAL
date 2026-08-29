"""Flickr8k dataset loading, splitting, and PyTorch Dataset classes."""
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.config import config
from src.vocabulary import Vocabulary


class Flickr8kCaptions:
    """Loads captions.txt and produces leakage-free train/val/test image-id splits.

    Flickr8k has 5 captions per image. Splitting by *caption* would leak the same
    image into both train and test, so we split by *image id* instead.
    """

    def __init__(self, captions_file: str = config.CAPTIONS_FILE, seed: int = config.SEED):
        self.df = pd.read_csv(captions_file)
        self.df.columns = [c.lower().strip() for c in self.df.columns]
        # normalize column names: expect "image" and "caption"
        if "image" not in self.df.columns:
            self.df = self.df.rename(columns={self.df.columns[0]: "image"})
        if "caption" not in self.df.columns:
            self.df = self.df.rename(columns={self.df.columns[1]: "caption"})
        self.df["image"] = self.df["image"].astype(str).str.strip()

        image_ids = sorted(self.df["image"].unique().tolist())
        rng = random.Random(seed)
        rng.shuffle(image_ids)

        n = len(image_ids)
        n_test = int(n * config.TEST_SPLIT)
        n_val = int(n * config.VAL_SPLIT)

        self.test_ids = set(image_ids[:n_test])
        self.val_ids = set(image_ids[n_test:n_test + n_val])
        self.train_ids = set(image_ids[n_test + n_val:])

    def split_df(self, split: str) -> pd.DataFrame:
        ids = {"train": self.train_ids, "val": self.val_ids, "test": self.test_ids}[split]
        return self.df[self.df["image"].isin(ids)].reset_index(drop=True)

    def build_vocab(self) -> Vocabulary:
        train_captions = self.split_df("train")["caption"].tolist()
        return Vocabulary(min_freq=config.MIN_WORD_FREQ).build(train_captions)


class ImageOnlyDataset(Dataset):
    """Used once, at feature-extraction time: yields (image_id, preprocessed image tensor)."""

    def __init__(self, image_ids, images_dir: str, transform):
        self.image_ids = sorted(set(image_ids))
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.images_dir, image_id)
        image = Image.open(path).convert("RGB")
        return image_id, self.transform(image)


class CaptionFeatureDataset(Dataset):
    """Training-time dataset: pairs a cached image-feature vector with one tokenized caption."""

    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, features_dir: str = config.FEATURES_DIR,
                 max_len: int = config.MAX_CAPTION_LEN):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.features_dir = features_dir
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        feat_path = os.path.join(self.features_dir, row["image"] + ".npy")
        feature = torch.from_numpy(np.load(feat_path)).float()
        caption_ids, length = self.vocab.encode(row["caption"], self.max_len)
        return {
            "feature": feature,
            "caption": torch.tensor(caption_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "image_id": row["image"],
        }
