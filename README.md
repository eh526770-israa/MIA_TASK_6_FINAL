# Task 1.1 — Modernizing the Machine Translation Model

English → French Seq2Seq translation, forked from the workshop notebook and upgraded with **pretrained word embeddings** and **proper MT evaluation metrics**.

## 1. Overall approach

The workshop notebook built an English→French translator with a from-scratch, randomly-initialized (frequency-based) embedding layer, a **bidirectional LSTM encoder**, **Luong (dot-product) attention**, and an **LSTM decoder** trained with teacher forcing.

This version keeps the same architecture (as required) and only changes **how words are turned into vectors**: instead of learning embeddings from zero, we start from **pretrained word vectors** and fine-tune them during training. On top of that, we replace "token accuracy" with real translation-quality metrics: **BLEU** and **ROUGE**.

## 2. Word-embedding method

- **English side:** GloVe (`glove.6B.300d.txt`, 300-dimensional, trained on Wikipedia + Gigaword).
- **French side:** FastText (`cc.fr.300.vec`, 300-dimensional, trained on Common Crawl + Wikipedia).

Both files use the same "word2vec text" line format (`word v1 v2 ... vn`), so one loader function (`load_pretrained_vectors`) handles both. For every word in our vocabulary:
- If the word exists in the pretrained file → its vector is copied in.
- If not (e.g. rare or misspelled words) → a small random vector is used instead.

The resulting matrix initializes `nn.Embedding.from_pretrained(matrix, freeze=False)`, so training still **fine-tunes** the vectors on our specific translation data — this generally beats both "fully frozen pretrained" and "fully random" on a small in-domain corpus.

## 3. BiLSTM encoder + attention (unchanged from the workshop)

- **Encoder:** a single-layer **bidirectional LSTM** reads the English sentence left→right and right→left, then concatenates both final states. This lets every position "see" both past and future context (important for words like *bank* whose meaning depends on both sides).
- **Attention (Luong/dot-product):** at every decoding step, the decoder's hidden state is compared (dot product) against all encoder outputs. A softmax turns those scores into weights, and a weighted sum of encoder outputs (the "context vector") tells the decoder which English words matter right now — instead of compressing the whole sentence into one fixed vector.
- **Decoder:** a single-layer LSTM that, at each step, combines its own hidden state with the attention context to predict the next French word. Trained with **teacher forcing** (fed the correct previous French word during training).

## 4. Preprocessing and training

1. **Cleaning:** lowercase, expand common contractions (`don't` → `do not`), strip punctuation/digits, collapse whitespace.
2. **Split:** 80% train / 10% validation / 10% test, split *before* building the vocabulary (no leakage).
3. **Vocabulary:** top-10,000 most frequent words per language + `<pad>`, `<unk>`, `<start>`, `<end>`.
4. **Embedding matrix:** built as described above (Section 2).
5. **Training:** Adam optimizer, cross-entropy loss with `ignore_index=<pad>` (so padding never inflates the loss/accuracy), gradient clipping (`max_norm=1.0`), `ReduceLROnPlateau` learning-rate scheduling, and **early stopping** on validation loss (best weights restored).

## 5. Evaluation metrics

- **BLEU** (`nltk.translate.bleu_score.corpus_bleu`, smoothing method 4): precision-based n-gram overlap between generated and reference French sentences — the standard MT benchmark metric.
- **ROUGE-1 / ROUGE-2 / ROUGE-L** (`rouge_score` library): recall-oriented overlap (unigram, bigram, and longest-common-subsequence), reported as F1. More forgiving of word reordering than BLEU, useful as a second opinion.

Both are computed on a sample of the held-out test set using **greedy decoding** (`translate_sentence`).

## 6. Results

*(fill in after running — the notebook prints and saves these automatically)*

| Metric | Score |
|---|---|
| Test loss | — |
| Test token accuracy (pad ignored) | — |
| BLEU | — |
| ROUGE-1 F1 | — |
| ROUGE-2 F1 | — |
| ROUGE-L F1 | — |

Add 2–3 sentences here discussing what you observed, e.g.: whether fine-tuned pretrained embeddings beat the workshop's from-scratch embeddings, which sentence types (short/common vs. long/rare) the model handles well, and what the attention heatmap shows.

## 7. Additional techniques used

- Fine-tuning (not freezing) the pretrained embeddings.
- Gradient clipping and `ReduceLROnPlateau` to stabilize BiLSTM training.
- Early stopping with best-weight restoration to avoid overfitting.
- Padding-aware loss/accuracy (`ignore_index`) so metrics aren't inflated by `<pad>` tokens.

## How to run

1. Get the dataset (`eng-fra.txt` or a Kaggle English–French CSV) and the pretrained vector files:
   - GloVe: https://nlp.stanford.edu/projects/glove/ (use `glove.6B.300d.txt`)
   - FastText French: https://fasttext.cc/docs/en/crawl-vectors.html (use `cc.fr.300.vec`)
2. On Kaggle: add both as "Input" datasets, then set `DATA_PATH`, `GLOVE_EN_PATH`, `FASTTEXT_FR_PATH` at the top of the notebook to their paths.
3. Run all cells top to bottom.
4. `pip install nltk rouge-score` if not already available in your environment.
