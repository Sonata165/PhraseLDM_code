import os
import sys

dirof = os.path.dirname
sys.path.append(dirof(dirof(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import T5Config
from models.phrase_vae import S2SVQAE


def main():
    # test_encode_decode_batch()
    test_forward()


def procedures():
    test_encode_decode()
    test_encode_decode_batch()
    test_forward()


class DummyVQ(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        # Identity quantization for test
        return x, torch.tensor(0.0), {}


def test_encode_decode():
    # Use a small T5 config for fast test
    tokenizer_path = "LongshenOu/phrase-vae-tokenizer"  # Should exist or mock
    from transformers import PreTrainedTokenizerFast

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        vocab_size = len(tokenizer)
    except Exception as e:
        print("Tokenizer loading failed (expected in CI):", e)
        return
    config = T5Config(
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_heads=2,
        vocab_size=vocab_size,
        decoder_start_token_id=tokenizer.bos_token_id,
    )
    t5_model_name = None  # Use random init for test
    vq = DummyVQ(hidden_size=32)
    model = S2SVQAE(t5_model_name, tokenizer_path, t5_config=config)
    print("--- Model initialized.")

    # Dummy input
    input_texts = ["o-0 p-172 d-1 o-6 p-19 d-2"]
    # Prepend [COMPRESS] token and tokenize
    input_texts = ["[COMPRESS] " + t for t in input_texts]
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Test encode
    encode_out = model.encode(input_ids, attention_mask=attention_mask)
    assert "quantized" in encode_out
    assert encode_out["quantized"].shape[0] == 1
    print("--- Encode test passed.")

    # Test decode
    quantized = encode_out["quantized"]  # [bs, code_size]
    generated_ids = model.decode(quantized, max_length=10)
    print("Generated IDs:", generated_ids)

    # Detokenize
    decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("Decoded text:", decoded_text)

    print("--- Decode test passed.")


def test_encode_decode_batch():
    # Use a small T5 config for fast test
    tokenizer_path = "LongshenOu/phrase-vae-tokenizer"  # Should exist or mock
    from transformers import PreTrainedTokenizerFast

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        vocab_size = len(tokenizer)
    except Exception as e:
        print("Tokenizer loading failed (expected in CI):", e)
        return
    config = T5Config(
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_heads=2,
        vocab_size=vocab_size,
    )
    t5_model_name = None  # Use random init for test
    model = S2SVQAE(
        t5_model_name,
        tokenizer_path,
        t5_config=config,
        vq_dim=32,
        vq_codebook_size=64,
        commitment_cost=0.25,
    )

    print("--- Model initialized.")

    # Dummy input
    input_texts = [
        "o-0 p-172 d-1 o-6 p-19 d-2",
        "o-0 p-172 d-1 o-6 p-19 d-2 o-6 p-19 d-2",
    ]
    print("Input texts:", input_texts)
    # Prepend [COMPRESS] token and tokenize
    input_texts = ["[COMPRESS] " + "[BOS]" + t + "[EOS]" for t in input_texts]
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    print("Input IDs:", input_ids)
    # detokenize to verify
    detok_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    print("Detokenized texts:", detok_texts)

    # Test encode
    encode_out = model.encode(input_ids, attention_mask=attention_mask)
    assert "quantized" in encode_out
    # assert encode_out['quantized'].shape[0] == 1
    print("--- Encode test passed.")

    # Test decode
    quantized = encode_out["quantized"]  # [bs, code_size]
    generated_ids = model.decode(quantized, max_length=10)
    print("Generated IDs:", generated_ids)

    # Detokenize
    decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    print("Decoded text:", decoded_text)

    print("--- Decode test passed.")


def test_forward():
    tokenizer_path = "LongshenOu/phrase-vae-tokenizer"
    from transformers import PreTrainedTokenizerFast

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        vocab_size = len(tokenizer)
    except Exception as e:
        print("Tokenizer loading failed (expected in CI):", e)
        return
    config = T5Config(
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_heads=2,
        vocab_size=vocab_size,
        decoder_start_token_id=tokenizer.bos_token_id,
    )
    print('Vocab size:', vocab_size)
    print('Decoder start token id:', config.decoder_start_token_id)
    t5_model_name = None
    model = S2SVQAE(
        t5_model_name,
        tokenizer_path,
        t5_config=config,
        vq_dim=32,
        vq_codebook_size=64,
        commitment_cost=0.25,
    )

    input_texts = [
        "o-0 i-128 p-172 d-1 b-1 o-6 i-33 p-19 d-2",
        "o-0 i-128 p-172 d-1 b-1 o-6",
    ]
    input_texts = ["[COMPRESS] " + "[BOS]" + t + "[EOS]" for t in input_texts]

    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    labels = input_ids.clone()[:, 1:]  # Remove [COMPRESS]
    labels = labels.contiguous()
    # Set padded tokens to -100 to ignore in loss
    labels[labels == tokenizer.pad_token_id] = -100
    print("Labels:", labels)

    # labels = torch.randint(0, vocab_size, (1, 10))
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    print("Forward test passed.")


if __name__ == "__main__":
    main()
