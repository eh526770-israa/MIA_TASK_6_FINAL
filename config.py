"""Central configuration for the Image Caption Generator project."""
import os
import torch


class Config:
    # ---- Paths ----
    DATA_DIR = os.environ.get("FLICKR8K_DIR", "data/flickr8k")
    IMAGES_DIR = os.path.join(DATA_DIR, "Images")
    CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")  # image,caption columns

    FEATURES_DIR = "artifacts/features"          # cached CNN features (one .npy per image)
    CHECKPOINT_DIR = "artifacts/checkpoints"
    VOCAB_PATH = "artifacts/vocab.pkl"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    # ---- Dataset ----
    VAL_SPLIT = 0.10
    TEST_SPLIT = 0.10
    MIN_WORD_FREQ = 3          # words seen fewer times become <unk>
    MAX_CAPTION_LEN = 35

    # ---- CNN feature extractor ----
    CNN_BACKBONE = "resnet50"   # "resnet50" | "inceptionv3" | "efficientnet_b0"
    IMAGE_SIZE = 224 if CNN_BACKBONE != "inceptionv3" else 299
    FEATURE_DIM = {"resnet50": 2048, "inceptionv3": 2048, "efficientnet_b0": 1280}[CNN_BACKBONE]

    # ---- Model ----
    EMBED_DIM = 256
    ATTENTION_DIM = 256
    DECODER_HIDDEN_DIM = 512
    DROPOUT = 0.5

    # ---- Training ----
    BATCH_SIZE = 8
    EPOCHS = 2
    LEARNING_RATE = 3e-4
    GRAD_CLIP = 5.0
    PATIENCE = 5                # early stopping
    LR_PATIENCE = 2             # ReduceLROnPlateau
    NUM_WORKERS = 0
    SEED = 42

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()
