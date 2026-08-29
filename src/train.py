"""Trains the caption model: early stopping, LR scheduling, checkpointing."""
import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import config
from src.dataset import Flickr8kCaptions, CaptionFeatureDataset
from src.model import CaptionModel
from tqdm import tqdm


class Trainer:
    def __init__(self, model, vocab, train_loader, val_loader, device=config.DEVICE):
        self.model = model.to(device)
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=config.LR_PATIENCE
        )
        self.history = {"train_loss": [], "val_loss": []}

    def _run_epoch(self, loader, train: bool):
        self.model.train() if train else self.model.eval()
        total_loss, n_batches = 0.0, 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in tqdm(loader, desc="training" if train else "validating", leave=False):
                features = batch["feature"].to(self.device)
                captions = batch["caption"].to(self.device)
                cap_in, cap_out = captions[:, :-1], captions[:, 1:]

                logits = self.model(features, cap_in)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), cap_out.reshape(-1))

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP)
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def fit(self, epochs=config.EPOCHS, patience=config.PATIENCE, checkpoint_dir=config.CHECKPOINT_DIR):
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val = math.inf
        stale = 0
        t0 = time.time()

        print(f"{'epoch':>6}  {'train_loss':>11}  {'val_loss':>9}  {'time':>6}")
        print("-" * 42)

        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            elapsed = time.time() - start
            print(f"{epoch:6d}  {train_loss:11.4f}  {val_loss:9.4f}  {elapsed:5.0f}s")

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                stale = 0
                torch.save({
                    "model_state": self.model.state_dict(),
                    "vocab_size": len(self.vocab),
                    "epoch": epoch,
                    "val_loss": best_val,
                }, os.path.join(checkpoint_dir, "best_model.pt"))
            else:
                stale += 1
                if stale >= patience:
                    print(f"Early stopping at epoch {epoch} (best val loss = {best_val:.4f})")
                    break

        print(f"\nFinished in {(time.time() - t0) / 60:.1f} min. Best val loss: {best_val:.4f}")
        return self.history


def main():
    torch.manual_seed(config.SEED)
    data = Flickr8kCaptions()
    vocab = data.build_vocab()
    vocab.save(config.VOCAB_PATH)

    train_ds = CaptionFeatureDataset(data.split_df("train"), vocab)
    val_ds = CaptionFeatureDataset(data.split_df("val"), vocab)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                               num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS)

    model = CaptionModel(vocab_size=len(vocab), pad_idx=vocab.pad_idx)
    trainer = Trainer(model, vocab, train_loader, val_loader)
    trainer.fit()


if __name__ == "__main__":
    main()
