"""Caption generation model: CNN features -> projection -> LSTM decoder.

Architecture (as specified in the task):
    Pretrained CNN -> Image Features -> LSTM -> Generated Caption

The CNN itself runs once offline (see feature_extractor.py) and features are
cached, so this module only contains the trainable part: a small projection
layer for the image feature, a word-embedding table, and an LSTM decoder.
"""
import torch
import torch.nn as nn

from src.config import config


class EncoderProjection(nn.Module):
    """Projects a cached CNN feature vector into the decoder's embedding space,
    and uses it to initialize the LSTM's hidden and cell state (Show-and-Tell style)."""

    def __init__(self, feature_dim=config.FEATURE_DIM, hidden_dim=config.DECODER_HIDDEN_DIM,
                 dropout=config.DROPOUT):
        super().__init__()
        self.init_hidden = nn.Linear(feature_dim, hidden_dim)
        self.init_cell = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor):
        features = self.dropout(features)
        h0 = torch.tanh(self.init_hidden(features)).unsqueeze(0)  # (1, B, H)
        c0 = torch.tanh(self.init_cell(features)).unsqueeze(0)
        return h0, c0


class DecoderRNN(nn.Module):
    """LSTM language model conditioned on the image feature via its initial state,
    plus a lightweight attention-style gate that re-weights the image feature at
    every decoding step (keeps the "attention mechanism" requirement even though
    the cached feature is a single pooled vector rather than a spatial map)."""

    def __init__(self, vocab_size, embed_dim=config.EMBED_DIM, hidden_dim=config.DECODER_HIDDEN_DIM,
                 feature_dim=config.FEATURE_DIM, pad_idx=0, dropout=config.DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.feature_gate = nn.Linear(hidden_dim + feature_dim, feature_dim)
        self.lstm = nn.LSTM(embed_dim + feature_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, captions_in: torch.Tensor, h0, c0, features: torch.Tensor):
        """captions_in: (B, T) token ids (teacher forcing input, includes <start>, excludes last token)
        features: (B, feature_dim) cached CNN features
        """
        B, T = captions_in.shape
        embedded = self.dropout(self.embedding(captions_in))  # (B, T, E)

        h, c = h0, c0
        outputs = []
        for t in range(T):
            gate = torch.sigmoid(self.feature_gate(torch.cat([h[-1], features], dim=-1)))
            gated_feature = gate * features                          # simple attention-like re-weighting
            lstm_input = torch.cat([embedded[:, t, :], gated_feature], dim=-1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            outputs.append(out.squeeze(1))
        outputs = torch.stack(outputs, dim=1)  # (B, T, H)
        logits = self.fc_out(self.dropout(outputs))
        return logits


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, pad_idx=0):
        super().__init__()
        self.encoder = EncoderProjection()
        self.decoder = DecoderRNN(vocab_size, pad_idx=pad_idx)

    def forward(self, features, captions_in):
        h0, c0 = self.encoder(features)
        return self.decoder(captions_in, h0, c0, features)

    @torch.no_grad()
    def generate(self, features: torch.Tensor, vocab, max_len=config.MAX_CAPTION_LEN):
        """Greedy decoding for a single image (features shape: (1, feature_dim))."""
        self.eval()
        h, c = self.encoder(features)
        token = torch.full((1, 1), vocab.sos_idx, dtype=torch.long, device=features.device)
        output_ids = []
        for _ in range(max_len):
            embedded = self.decoder.embedding(token)
            gate = torch.sigmoid(self.decoder.feature_gate(torch.cat([h[-1], features], dim=-1)))
            gated_feature = gate * features
            lstm_input = torch.cat([embedded[:, 0, :], gated_feature], dim=-1).unsqueeze(1)
            out, (h, c) = self.decoder.lstm(lstm_input, (h, c))
            logits = self.decoder.fc_out(out.squeeze(1))
            next_id = int(logits.argmax(dim=-1))
            if next_id == vocab.eos_idx:
                break
            output_ids.append(next_id)
            token = torch.tensor([[next_id]], dtype=torch.long, device=features.device)
        return vocab.decode(output_ids)
