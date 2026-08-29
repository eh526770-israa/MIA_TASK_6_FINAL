import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.model import CaptionModel
from src.vocabulary import Vocabulary


def build_tiny_vocab():
    return Vocabulary(min_freq=1).build(["a dog runs in the park", "a cat sleeps on the mat"])


def test_forward_output_shape():
    vocab = build_tiny_vocab()
    model = CaptionModel(vocab_size=len(vocab), pad_idx=vocab.pad_idx)

    batch_size, seq_len, feature_dim = 4, 6, 2048
    features = torch.randn(batch_size, feature_dim)
    captions_in = torch.randint(0, len(vocab), (batch_size, seq_len))

    logits = model(features, captions_in)
    assert logits.shape == (batch_size, seq_len, len(vocab))


def test_generate_returns_string():
    vocab = build_tiny_vocab()
    model = CaptionModel(vocab_size=len(vocab), pad_idx=vocab.pad_idx)
    feature = torch.randn(1, 2048)

    caption = model.generate(feature, vocab, max_len=10)
    assert isinstance(caption, str)
