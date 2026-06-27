# %%
# Lets tokenise
import os
import tiktoken
import regex as re
import json
import sentencepiece as spm


with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

len(text)
text[:100]
# %%
# 0
string = "hello, this is a random piece of text :D except now im gonna extend it. Why? cos i realised that a short piece of text wasn't really gonna demonstrate anything. Idk why im writing this myself by hand instead of GPT or smth."
tokens = list(map(int, string.encode("utf-8")))
print(tokens)

# %%
# 0


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# %%
# 0


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


# %%
# 0

vocab_size = 276
num_merges = vocab_size - 256
ids = list(tokens)
merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    idx = 256 + i
    top_pair = max(stats, key=stats.get)
    print(f"Merging {top_pair} into {idx}")
    ids = merge(ids, top_pair, idx)
    merges[top_pair] = idx


# %%
# 0
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
print(vocab)


# %%
# 0
def decode(ids):
    out = ""
    for i in ids:
        out += vocab[i].decode("utf-8", errors="replace")

    return out


decode([128])

# %%
# 0


def encode(text):
    tokens = list(text.encode("utf-8"))
    while True:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


out = encode("Hello World!")
out = decode(out)

# %%
# 0

gpt2pat = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
print(re.findall(gpt2pat, "Hello, world! :D"))

# %%
# 0

enc = tiktoken.get_encoding("gpt2")
print(enc.encode("        hello world!!!"))

enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode("        hello world!!!"))

# %%
# 0

with open("encoder.json", "r") as f:
    encoder = json.load(f)
with open("vocab.bpe", "r", encoding="utf-8") as f:
    bpe_data = f.read()
bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]

# %%
# 0

with open("toy.txt", "w", encoding="utf-8") as f:
    f.write(
        "hello, this is a random piece of text :D except now im gonna extend it. Why? cos i realised that a short piece of text wasn't really gonna demonstrate anything. Idk why im writing this myself by hand instead of GPT or smth."
    )

# %%
# 0

options = dict(
    # input spec
    input="toy.txt",
    input_format="text",
    # output spec
    model_prefix="tok400",  # output filename prefix
    # algorithm spec
    # BPE alg
    model_type="bpe",
    vocab_size=400,
    # normalization
    normalization_rule_name="identity",  # ew, turn off normalization
    remove_extra_whitespaces=False,
    input_sentence_size=200000000,  # max number of training sentences
    max_sentence_length=4192,  # max number of bytes per sentence
    seed_sentencepiece_size=1000000,
    shuffle_input_sentence=True,
    # rare word treatment
    character_coverage=0.99995,
    byte_fallback=True,
    # merge rules
    split_digits=True,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    max_sentencepiece_length=16,
    add_dummy_prefix=True,
    allow_whitespace_only_pieces=True,
    # special tokens
    unk_id=0,  # the UNK token MUST exist
    bos_id=1,  # the others are optional, set to -1 to turn off
    eos_id=2,
    pad_id=-1,
    # systems
    num_threads=os.cpu_count(),  # use ~all system resources
)
spm.SentencePieceTrainer.train(**options)

# %%
# 0

sp = spm.SentencePieceProcessor()
sp.load("tok400.model")
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]
vocab
