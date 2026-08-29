# Task 1.2 — Image Caption Generator (Flickr8k)

An end-to-end, production-oriented Image Caption Generator: given a photo, the model generates a natural-language description. Built as reusable modules (not a single notebook), trained with transfer learning, evaluated with standard metrics, and deployed behind a simple UI.

## 1. Project overview

Given an image, the system extracts visual features with a **pretrained CNN**, then generates a caption word-by-word with an **LSTM decoder** conditioned on those features. Trained and evaluated on the **Flickr8k dataset** (8,000 images, 5 human-written captions each).

## 2. Dataset

- **Flickr8k**: 8,000 images, 5 reference captions per image.
- Splits are done **by image id** (90/10/... see `TEST_SPLIT`/`VAL_SPLIT` in `src/config.py`) so all 5 captions of one image always stay in the same split — this avoids the data leakage you'd get by splitting on individual captions.
- Download the dataset (e.g. from Kaggle: "Flickr8k Dataset") and arrange it as:
```
data/flickr8k/
  Images/           # all .jpg files
  captions.txt       # columns: image, caption
```

## 3. Architecture

```
Image → Pretrained CNN (ResNet50, frozen) → 2048-d feature vector
                                                   │
                                    projected to init LSTM hidden/cell state
                                                   │
<start> → Embedding → LSTM (feature re-injected via a gate at every step) → next word
                                                   │  (repeat until <end>)
                                              Generated Caption
```

- **Image encoder**: pretrained **ResNet50** (ImageNet weights) with the classification head removed — pure **transfer learning**, the CNN itself is frozen and never fine-tuned in this version.
- **Decoder**: single-layer **LSTM**. The image feature initializes the LSTM's hidden/cell state (Show-and-Tell style) *and* is re-weighted by a small learned gate and re-injected at every decoding step, so the model keeps "looking at" the image while generating each word (a lightweight attention-style mechanism).
- **Text**: standard `<start>` / `<end>` / `<pad>` / `<unk>` vocabulary, built only from the training split.

## 4. Preprocessing

- **Images**: resized to 224×224, normalized with ImageNet mean/std (`src/feature_extractor.py`). Features are extracted **once** and cached as `.npy` files (`artifacts/features/`), so training epochs don't re-run the CNN — much faster.
- **Captions**: lowercased, punctuation stripped, tokenized on whitespace, wrapped with `<start>`/`<end>`, padded to `MAX_CAPTION_LEN`. Words seen fewer than `MIN_WORD_FREQ` times become `<unk>`.

## 5. Training process

- Adam optimizer, cross-entropy loss with `ignore_index=<pad>`.
- Gradient clipping, `ReduceLROnPlateau` learning-rate scheduling, and **early stopping** on validation loss.
- Best checkpoint (lowest val loss) is saved automatically to `artifacts/checkpoints/best_model.pt`.

## 6. Evaluation metrics and results

Evaluated on the held-out **test** split (unseen images), against all 5 references per image:
- **BLEU** — n-gram precision, the standard MT/captioning metric.
- **ROUGE-1/2/L (F1)** — n-gram / longest-common-subsequence recall-oriented overlap.
- **METEOR** — accounts for synonyms/stemming, generally correlates better with human judgment than BLEU alone.

 
 

**Qualitative examples** (Input Image → Generated Caption → Reference Captions) are also printed by `src/evaluate.py`; paste a few screenshots or a short recording here as required by the task.

## 7. Software engineering

The project is organized into reusable modules instead of one notebook:

```
image_caption_project/
├── src/
│   ├── config.py             # all paths & hyperparameters in one place
│   ├── vocabulary.py         # tokenization + vocab build/encode/decode
│   ├── dataset.py            # leakage-free split + PyTorch Datasets
│   ├── feature_extractor.py  # pretrained CNN, feature caching
│   ├── model.py               # EncoderProjection + DecoderRNN + CaptionModel
│   ├── train.py               # Trainer class: fit(), early stopping, checkpoints
│   ├── evaluate.py            # BLEU / ROUGE / METEOR + qualitative examples
│   └── predict.py             # single-image inference used by app/API
├── tests/
│   ├── test_vocabulary.py
│   └── test_model.py
├── app.py                     # Streamlit UI
├── api.py                     # optional FastAPI service
├── requirements.txt
├── Dockerfile
└── README.md
```

Run tests with:
```bash
pytest tests/ -v
```

## 8. Production / deployment

Two ready-to-run options are included:

- **Streamlit UI** (`app.py`) — upload an image, see the generated caption.
- **FastAPI service** (`api.py`) — `POST /caption` with an image file, get `{"caption": "..."}` back.

### Docker
```bash
docker build -t image-caption-app .
docker run -p 8501:8501 image-caption-app
 
```

## 9. Model storage and sharing

After training, upload `artifacts/checkpoints/best_model.pt` and `artifacts/vocab.pkl` to a HuggingFace model repo (`huggingface-cli upload`), then put the link here:

> Model: `https://huggingface.co/<your-username>/<repo-name>`

## How to run — full pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put the Flickr8k dataset at data/flickr8k/ (Images/ + captions.txt)

# 3. Extract and cache CNN features (run once)
python -m src.feature_extractor

# 4. Train
python -m src.train

# 5. Evaluate on the test set (BLEU/ROUGE/METEOR + qualitative examples)
python -m src.evaluate

# 6. Try it
streamlit run app.py
# or
uvicorn api:app --reload
```
