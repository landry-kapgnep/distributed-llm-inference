"""
Visual demo.
Runs everything in only one process for clarity purposes.
No need of gRPC servers.
"""
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

from src.stage import GPT2Stage


def main():
    model_name = "gpt2"
    prompt = "The meaning of life is"
    n_stages = 2
    max_tokens = 8

    print("=" * 60)
    print("  PIPELINE-PARALLEL INFERENCE DEMO")
    print("=" * 60)

    # ── 1. Show the full model ──────────────────────────────────

    print("\n[1] Loading GPT-2 124M...\n")
    full = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    n_params = sum(p.numel() for p in full.parameters())
    n_blocks = len(full.transformer.h)
    hidden = full.config.n_embd
    vocab = full.config.vocab_size

    print(f"    Parameters:  {n_params:,}")
    print(f"    Blocks:      {n_blocks} transformer layers")
    print(f"    Hidden size: {hidden}")
    print(f"    Vocabulary:  {vocab:,} tokens")

    del full

    # ── 2. Split into stages ────────────────────────────────────

    print(f"\n[2] Splitting model into {n_stages} stages\n")
    stages = []

    for i in range(n_stages):
        stage = GPT2Stage(model_name, i, n_stages)
        params = sum(p.numel() for p in stage.parameters())

        parts = []
        if stage.is_first:
            parts += ["token embedding (wte)", "position embedding (wpe)"]
        parts.append(f"transformer blocks [{stage.layer_start}..{stage.layer_end - 1}]")
        if stage.is_last:
            parts += ["layer norm (ln_f)", "lm_head (768 -> 50257)"]

        label = f"Worker {i}  (simulates machine on port {50051 + i})"
        print(f"    {label}")
        for p in parts:
            print(f"      - {p}")
        print(f"      ({params:,} parameters)\n")

        stages.append(stage)

    # ── 3. Tokenize ─────────────────────────────────────────────

    print(f'[3] Tokenizing: "{prompt}"\n')
    ids = tokenizer.encode(prompt, return_tensors="pt")
    tokens = [tokenizer.decode(t) for t in ids[0]]

    print(f"    Tokens: {tokens}")
    print(f"    IDs:    {ids[0].tolist()}")
    print(f"    Shape:  {list(ids.shape)}  (batch=1, seq_len={ids.shape[1]})")

    # ── 4. Generate ─────────────────────────────────────────────

    print(f"\n[4] Generating {max_tokens} tokens through the pipeline\n")
    print("-" * 60)

    for step in range(max_tokens):
        print(f"\n  --- token {step + 1}/{max_tokens} ---\n")
        x = ids

        for i, stage in enumerate(stages):
            in_shape = list(x.shape)
            in_bytes = x.nelement() * x.element_size()

            x = stage(x)

            out_shape = list(x.shape)
            out_bytes = x.nelement() * x.element_size()

            if stage.is_first:
                print(f"  >> Worker {i}:")
                print(f"     input:  token IDs {in_shape}  ({in_bytes:,} bytes)")
                print(f"     embed:  {in_shape[0]}x{in_shape[1]} tokens -> [{in_shape[0]}, {in_shape[1]}, 768]")
                print(f"     blocks: self-attention + FFN x{stage.layer_end - stage.layer_start}")
                print(f"     output: hidden states {out_shape}  ({out_bytes:,} bytes)")
            else:
                print(f"  >> Worker {i}:")
                print(f"     input:  hidden states {in_shape}  ({in_bytes:,} bytes)")
                print(f"     blocks: self-attention + FFN x{stage.layer_end - stage.layer_start}")
                print(f"     ln_f:   layer norm")
                print(f"     head:   project [768] -> [{vocab}] for last position")
                print(f"     output: logits {out_shape}  ({out_bytes:,} bytes)")

            if i < len(stages) - 1:
                print(f"\n  ~~ gRPC transfer: {out_bytes:,} bytes over network ~~\n")

        probs = torch.softmax(x[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, 1)
        token_str = tokenizer.decode(next_id[0])

        ids = torch.cat([ids, next_id], dim=1)

        generated_so_far = tokenizer.decode(ids[0])
        print(f'\n  -> sampled "{token_str}" (id={next_id.item()})')
        print(f'  -> sequence so far: "{generated_so_far}"')

    # ── 5. Done ─────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print(f'  Result: "{tokenizer.decode(ids[0])}"')
    print("=" * 60)


if __name__ == "__main__":
    main()
