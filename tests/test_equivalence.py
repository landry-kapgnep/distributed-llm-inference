import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

from src.stage import GPT2Stage


def test_split_matches_original():
    model_name = "gpt2"
    n_stages = 2

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ids = tokenizer.encode("Hello world", return_tensors="pt")

    full = GPT2LMHeadModel.from_pretrained(model_name)
    full.eval()
    with torch.no_grad():
        expected = full(ids).logits[:, -1:, :]

    stages = [GPT2Stage(model_name, i, n_stages) for i in range(n_stages)]
    x = ids
    for stage in stages:
        x = stage(x)

    assert x.shape == expected.shape, f"{x.shape} != {expected.shape}"
    assert torch.allclose(x, expected, atol=1e-5), f"max diff: {(x - expected).abs().max()}"


def test_three_way_split():
    model_name = "gpt2"
    n_stages = 3

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ids = tokenizer.encode("The quick brown fox", return_tensors="pt")

    full = GPT2LMHeadModel.from_pretrained(model_name)
    full.eval()
    with torch.no_grad():
        expected = full(ids).logits[:, -1:, :]

    stages = [GPT2Stage(model_name, i, n_stages) for i in range(n_stages)]
    x = ids
    for stage in stages:
        x = stage(x)

    assert x.shape == expected.shape
    assert torch.allclose(x, expected, atol=1e-5)
