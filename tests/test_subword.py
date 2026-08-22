"""babble/subword.py: BPE and word-level tokenizers, experimental and
off the production path (see babble/tokenizer.py for the one that ships)."""

from __future__ import annotations

from babble import tokenizer as bytetok
from babble.subword import (
    BPETokenizer,
    ByteTokenizer,
    WordTokenizer,
    build_continuation_example,
    build_text_example,
    stack_examples,
    text_context,
    text_examples,
)

CORPUS = [
    "hello there friend",
    "hello again my friend",
    "the cat sat on the mat",
    "the cat and the hat",
    "why is the sky blue",
    "why is the cat here",
    "boop the snoot",
    "boop boop boop",
]


# --- BPE --------------------------------------------------------------


def test_bpe_learns_the_requested_number_of_merges_or_fewer():
    tok = BPETokenizer.train(CORPUS, num_merges=20)
    assert len(tok.merges) <= 20
    assert len(tok.merges) > 0


def test_bpe_merges_never_cross_a_whitespace_boundary():
    # "e " (end of "the", start of a space) is common in CORPUS; if merges
    # crossed chunk boundaries this pair would be a prime merge candidate.
    tok = BPETokenizer.train(CORPUS, num_merges=50)
    for a, b, _ in tok.merges:
        pa, pb = tok.vocab[a], tok.vocab[b]
        # Neither half of a merge may straddle a whitespace/non-whitespace
        # transition -- each merge's combined bytes decode to text that is
        # either all-whitespace or all-non-whitespace.
        combined = (pa + pb).decode("utf-8", errors="replace")
        assert combined.isspace() or not any(c.isspace() for c in combined)


def test_bpe_round_trips_training_text():
    tok = BPETokenizer.train(CORPUS, num_merges=100)
    for text in CORPUS:
        assert tok.decode(tok.encode(text)) == text


def test_bpe_round_trips_unseen_text_too():
    # Total function: an OOV string still decodes byte-for-byte, same
    # guarantee babble/tokenizer.py makes for raw bytes.
    tok = BPETokenizer.train(CORPUS, num_merges=100)
    text = "a completely unseen sentence with 👍 emoji and a URL http://x.io"
    assert tok.decode(tok.encode(text)) == text


def test_bpe_shrinks_the_token_count_vs_bytes():
    tok = BPETokenizer.train(CORPUS, num_merges=100)
    text = "the cat and the hat"
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_bpe_zero_merges_is_pure_bytes():
    tok = BPETokenizer.train(CORPUS, num_merges=0)
    text = "hello"
    assert tok.encode(text) == list(text.encode("utf-8"))


def test_bpe_specials_sit_above_the_learned_vocab():
    tok = BPETokenizer.train(CORPUS, num_merges=30)
    base = 256 + len(tok.merges)
    assert tok.specials.pad == base
    assert tok.specials.bos == base + 1
    assert tok.specials.sep == base + 2
    assert tok.specials.eos == base + 3
    assert tok.vocab_size == base + 4


def test_bpe_to_json_from_json_round_trips(tmp_path):
    """Same schema `pretrain_hf.py` writes: only `merges` needs to round-trip,
    `vocab` is rebuilt from replaying them -- this is what lets
    `post_train_from_checkpoint` load a tokenizer.json a pretrain job produced
    with no translation step."""
    tok = BPETokenizer.train(CORPUS, num_merges=15)
    path = tmp_path / "tokenizer.json"
    tok.to_json(path)
    reloaded = BPETokenizer.from_json(path)
    assert reloaded.merges == tok.merges
    assert reloaded.vocab == tok.vocab
    assert reloaded.vocab_size == tok.vocab_size
    for text in CORPUS:
        assert reloaded.encode(text) == tok.encode(text)


# --- word-level ---------------------------------------------------------


def test_word_tokenizer_gives_frequent_chunks_their_own_id():
    tok = WordTokenizer.train(CORPUS, max_words=50)
    assert "the" in tok.vocab
    assert tok.encode("the") == [tok.vocab["the"]]


def test_word_tokenizer_falls_back_to_bytes_for_unseen_words():
    tok = WordTokenizer.train(CORPUS, max_words=3)  # tiny cap, most words excluded
    ids = tok.encode("zzzznotinvocab")
    assert ids == list("zzzznotinvocab".encode("utf-8"))


def test_word_tokenizer_round_trips():
    tok = WordTokenizer.train(CORPUS, max_words=50)
    for text in CORPUS + ["something completely unseen 👍"]:
        assert tok.decode(tok.encode(text)) == text


def test_word_tokenizer_shrinks_token_count_for_known_words():
    tok = WordTokenizer.train(CORPUS, max_words=50)
    text = "the cat and the hat"
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_word_tokenizer_specials_sit_above_the_learned_vocab():
    tok = WordTokenizer.train(CORPUS, max_words=10)
    base = 256 + len(tok.vocab)
    assert tok.specials.eos == base + 3
    assert tok.vocab_size == base + 4


# --- shared example layout ----------------------------------------------


def test_text_examples_mask_starts_at_zero_and_trains_the_rest():
    tok = BPETokenizer.train(CORPUS, num_merges=20)
    examples = text_examples(tok, "hello there friend", block_size=32)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.tokens[0] == tok.specials.bos
    assert ex.tokens[-1] == tok.specials.eos
    assert ex.mask[0] == 0
    assert all(m == 1 for m in ex.mask[1:])


def test_text_examples_splits_long_rows_into_chunks_without_dropping_bytes():
    tok = BPETokenizer.train(CORPUS, num_merges=0)  # pure bytes: easy to count
    text = "boop " * 40
    block_size = 16
    examples = text_examples(tok, text, block_size)
    assert len(examples) > 1
    # Only the final chunk carries <eos>.
    assert examples[-1].tokens[-1] == tok.specials.eos
    for ex in examples[:-1]:
        assert tok.specials.eos not in ex.tokens
    # Reassembling the encoded ids from every chunk recovers the full text.
    recovered_ids = []
    for ex in examples:
        body = ex.tokens[1:]
        if body and body[-1] == tok.specials.eos:
            body = body[:-1]
        recovered_ids.extend(body)
    assert tok.decode(recovered_ids) == text


def test_text_examples_empty_text_is_no_examples():
    tok = BPETokenizer.train(CORPUS, num_merges=10)
    assert text_examples(tok, "", block_size=32) == []


def test_build_text_example_final_false_omits_eos():
    tok = BPETokenizer.train(CORPUS, num_merges=10)
    ex = build_text_example(tok, tok.encode("hello"), final=False)
    assert tok.specials.eos not in ex.tokens


# --- byte adapter matches production babble.tokenizer exactly ----------


def test_byte_tokenizer_ids_match_production_specials():
    tok = ByteTokenizer()
    assert tok.specials.pad == bytetok.PAD_ID
    assert tok.specials.bos == bytetok.BOS_ID
    assert tok.specials.sep == bytetok.SEP_ID
    assert tok.specials.eos == bytetok.EOS_ID
    assert tok.vocab_size == bytetok.VOCAB_SIZE


def test_byte_tokenizer_examples_match_production_text_examples():
    tok = ByteTokenizer()
    for text in CORPUS + ["boop " * 40]:
        for block_size in (16, 32, 64):
            got = text_examples(tok, text, block_size)
            want = bytetok.text_examples(text, block_size)
            assert [(e.tokens, e.mask) for e in got] == [(e.tokens, e.mask) for e in want]


def test_text_context_keeps_the_tail_under_budget():
    tok = BPETokenizer.train(CORPUS, num_merges=0)
    block_size = 12
    ctx = text_context(tok, "a very long prefix that will not fit", block_size)
    assert ctx[0] == tok.specials.bos
    assert len(ctx) <= block_size
    # Keeps the tail, matching babble.tokenizer.text_context's rule.
    assert tok.decode(ctx[1:]).endswith("fit")


# --- generalised continuation example / stacking, used by stage 2 against an
# externally pretrained (non-byte) tokenizer -- see posttrain.post_train_from_checkpoint


def test_build_continuation_example_masks_the_prefix_only():
    tok = BPETokenizer.train(CORPUS, num_merges=10)
    ex = build_continuation_example(tok, "the cat sat", "on the mat", block_size=32)
    assert ex.tokens[0] == tok.specials.bos
    assert ex.tokens[-1] == tok.specials.eos
    assert ex.mask[0] == 0  # <bos> is never trained
    assert ex.mask[-1] == 1  # <eos> is trained, matching babble.tokenizer's layout


def test_stack_examples_pads_with_the_tokenizer_own_pad_id_not_256():
    """A BPE tokenizer's first learned merge can land exactly on id 256 --
    the byte tokenizer's hardcoded PAD_ID. Padding must use the tokenizer's
    own pad id (above its learned vocab), never a bare 256."""
    tok = BPETokenizer.train(CORPUS, num_merges=3)
    assert tok.specials.pad != 256  # sanity: this tokenizer's pad is NOT the byte PAD_ID
    short = build_text_example(tok, tok.encode("hi"))
    long = build_text_example(tok, tok.encode("hello there friend, a much longer one"))
    tokens, mask, weights = stack_examples([short, long], tok.specials.pad)
    pad_positions = tokens[0, len(short.tokens):]
    assert (pad_positions == tok.specials.pad).all()
    assert not (pad_positions == 256).all()  # would be true if this used the byte PAD_ID by mistake
