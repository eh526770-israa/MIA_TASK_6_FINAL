import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vocabulary import Vocabulary


def test_special_tokens_present():
    vocab = Vocabulary()
    assert vocab.word2idx[Vocabulary.PAD] == 0
    assert vocab.word2idx[Vocabulary.SOS] == 1
    assert vocab.word2idx[Vocabulary.EOS] == 2
    assert vocab.word2idx[Vocabulary.UNK] == 3


def test_build_respects_min_freq():
    captions = ["a dog runs", "a dog barks", "a cat sleeps"]
    vocab = Vocabulary(min_freq=2).build(captions)
    assert "dog" in vocab.word2idx        # appears twice
    assert "a" in vocab.word2idx          # appears three times
    assert "cat" not in vocab.word2idx    # appears once -> below min_freq
    assert "sleeps" not in vocab.word2idx


def test_encode_decode_roundtrip():
    vocab = Vocabulary(min_freq=1).build(["a dog runs fast"])
    ids, length = vocab.encode("a dog runs", max_len=10)
    assert length == 5  # <start> a dog runs <end>
    assert len(ids) == 10
    decoded = vocab.decode(ids)
    assert decoded == "a dog runs"


def test_unknown_word_maps_to_unk():
    vocab = Vocabulary(min_freq=1).build(["a dog runs"])
    ids, _ = vocab.encode("a spaceship flies", max_len=10)
    words = [vocab.idx2word[i] for i in ids]
    assert Vocabulary.UNK in words
