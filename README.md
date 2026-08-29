# MIA Training '27 — Task 6 (AI Team)

This repository contains the submission for Task 6, covering both parts:

## 📁 Task 1.1 — Modernizing the Machine Translation Model
Path: `Task1.1_Machine_Translation/`

I forked the original workshop notebook and replaced the frequency-based word representation with **pretrained word embeddings**:
- **GloVe** for English (300d)
- **FastText** for French (300d)

while keeping the same core architecture as required:
- **BiLSTM Encoder**
- **Luong (dot-product) Attention**
- **LSTM Decoder** with teacher forcing

I also added proper machine-translation evaluation on top of the workshop's token-accuracy metric:
- **BLEU**
- **ROUGE-1 / ROUGE-2 / ROUGE-L**

Full details on the approach, preprocessing, and results are in `Task1.1_Machine_Translation/README.md`.

---

## 📁 Task 1.2 — Image Caption Generator (Flickr8k)
Path: `Task1.2_Image_Caption_Generator/`

I built a production-oriented end-to-end project that generates natural-language captions for images, using:
- **Transfer learning** with a pretrained ResNet50 to extract image features
- An **LSTM decoder** to generate captions word by word
- Evaluation using **BLEU / ROUGE / METEOR**
- A simple user interface via **Streamlit** (`app.py`) and an alternative **FastAPI** service (`api.py`)
- Code organized into reusable modules (`src/`) instead of a single large notebook
- Unit tests in `tests/`

Full details on the architecture, training process, and results are in `Task1.2_Image_Caption_Generator/README.md`.

---

## ⚠️ Note on the Git upload process

While uploading the Image Caption Generator project, I ran into an issue where my first commit accidentally included all Flickr8k images (8,000+ files), because I had committed before setting up `.gitignore` correctly. This resulted in a very large push (over 1 GB), which caused the upload to fail with a connection timeout (HTTP 408).

**How I fixed it:**
1. Deleted the entire local Git history (the `.git` folder) and reinitialized the repository from scratch.
2. Added a proper `.gitignore` that excludes:
   - `data/` (the dataset and images — too large and not meant to be version-controlled)
   - `venv/` (local Python virtual environment)
   - `__pycache__/`, `*.pyc`, `.pytest_cache/`
   - Trained model artifacts (`*.pt`, `*.npy`)
3. Re-committed the project without these files, which made the push fast and successful.

**Note:** The dataset (Flickr8k) and the trained model checkpoint are intentionally not included in this repository — they need to be available locally on any machine that runs the project. Setup instructions are provided in the Task 1.2 README.

---

## 👤 Submitted by
Israa — AI Team, Training '27
