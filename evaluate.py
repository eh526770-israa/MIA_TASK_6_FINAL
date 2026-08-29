"""Evaluates a trained caption model on unseen test images: BLEU, ROUGE, METEOR
plus qualitative Input Image -> Generated Caption -> Reference Captions examples."""
import numpy as np
import torch
from torch.utils.data import DataLoader

import nltk
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from src.config import config
from src.dataset import Flickr8kCaptions
from src.model import CaptionModel
from src.vocabulary import Vocabulary


class Evaluator:
    def __init__(self, model, vocab, device=config.DEVICE):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    @torch.no_grad()
    def caption_image(self, feature: np.ndarray) -> str:
        feat_t = torch.from_numpy(feature).float().unsqueeze(0).to(self.device)
        return self.model.generate(feat_t, self.vocab)

    def evaluate(self, df, features_dir=config.FEATURES_DIR, n_samples=None, seed=config.SEED):
        """df must have one row per image (use references grouped by image for fair BLEU/ROUGE)."""
        grouped = df.groupby("image")["caption"].apply(list).reset_index()
        if n_samples:
            grouped = grouped.sample(n=min(n_samples, len(grouped)), random_state=seed).reset_index(drop=True)

        references, hypotheses = [], []
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        meteor_scores = []
        qualitative = []

        for _, row in grouped.iterrows():
            image_id = row["image"]
            refs = row["caption"]
            feature = np.load(f"{features_dir}/{image_id}.npy")
            pred = self.caption_image(feature)

            hypotheses.append(pred.split())
            references.append([r.lower().split() for r in refs])

            best_rouge = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
            for ref in refs:
                scores = self.rouge.score(ref, pred)
                for k in best_rouge:
                    best_rouge[k] = max(best_rouge[k], scores[k].fmeasure)
            for k in rouge_scores:
                rouge_scores[k].append(best_rouge[k])

            meteor_scores.append(meteor_score([r.lower().split() for r in refs], pred.split()))
            qualitative.append({"image": image_id, "prediction": pred, "references": refs})

        smoothie = SmoothingFunction().method4
        bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)

        results = {"BLEU": bleu, "METEOR": float(np.mean(meteor_scores))}
        for k, v in rouge_scores.items():
            results[f"{k}_F1"] = float(np.mean(v))
        return results, qualitative


def main():
    data = Flickr8kCaptions()
    vocab = Vocabulary.load(config.VOCAB_PATH)
    test_df = data.split_df("test")

    ckpt = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model = CaptionModel(vocab_size=ckpt["vocab_size"], pad_idx=vocab.pad_idx)
    model.load_state_dict(ckpt["model_state"])

    evaluator = Evaluator(model, vocab)
    results, qualitative = evaluator.evaluate(test_df)

    print("Test-set caption quality:")
    for k, v in results.items():
        print(f"  {k:10s}: {v:.4f}")

    print("\nQualitative examples:")
    for ex in qualitative[:5]:
        print(f"\nImage: {ex['image']}")
        print(f"  Generated : {ex['prediction']}")
        for r in ex["references"][:2]:
            print(f"  Reference : {r}")


if __name__ == "__main__":
    main()
