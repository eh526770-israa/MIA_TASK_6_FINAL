"""Caption tokenization, vocabulary building, and numericalization."""
import re
import pickle
from collections import Counter


class Vocabulary:
    """Word-level vocabulary with special tokens for image captioning."""

    PAD, SOS, EOS, UNK = "<pad>", "<start>", "<end>", "<unk>"

    def __init__(self, min_freq: int = 3):
        self.min_freq = min_freq
        self.word2idx = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.pad_idx, self.sos_idx, self.eos_idx, self.unk_idx = 0, 1, 2, 3

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def tokenize(text: str):
        text = str(text).lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def build(self, captions):
        counts = Counter()
        for cap in captions:
            counts.update(self.tokenize(cap))
        for word, freq in counts.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        return self

    def encode(self, text: str, max_len: int):
        tokens = [self.SOS] + self.tokenize(text) + [self.EOS]
        ids = [self.word2idx.get(t, self.unk_idx) for t in tokens][:max_len]
        length = len(ids)
        ids = ids + [self.pad_idx] * (max_len - length)
        return ids, length

    def decode(self, ids, skip_special: bool = True):
        special = {self.PAD, self.SOS, self.EOS} if skip_special else set()
        words = []
        for i in ids:
            word = self.idx2word.get(int(i), self.UNK)
            if word in special:
                if word == self.EOS:
                    break
                continue
            words.append(word)
        return " ".join(words)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Vocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)
