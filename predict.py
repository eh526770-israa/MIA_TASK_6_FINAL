"""Loads a trained checkpoint once and exposes a simple predict(image) function,
used by both app.py (Streamlit UI) and any API wrapper."""
from PIL import Image
import torch

from src.config import config
from src.feature_extractor import CNNFeatureExtractor
from src.model import CaptionModel
from src.vocabulary import Vocabulary


class CaptionPredictor:
    def __init__(self, checkpoint_path: str = config.BEST_MODEL_PATH,
                 vocab_path: str = config.VOCAB_PATH, device=config.DEVICE):
        self.device = device
        self.vocab = Vocabulary.load(vocab_path)

        self.extractor = CNNFeatureExtractor().to(device)
        self.extractor.eval()
        self.transform = CNNFeatureExtractor.preprocess_transform()

        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model = CaptionModel(vocab_size=ckpt["vocab_size"], pad_idx=self.vocab.pad_idx)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        feature = self.extractor(tensor)
        return self.model.generate(feature, self.vocab)


# Lazily-created singleton so the (slow) model load happens only once per process.
_predictor = None


def get_predictor() -> CaptionPredictor:
    global _predictor
    if _predictor is None:
        _predictor = CaptionPredictor()
    return _predictor
