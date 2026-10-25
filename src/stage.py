import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class GPT2Stage(nn.Module):

    def __init__(self, model_name: str, stage_id: int, n_stages: int):
        super().__init__()
        self.model_name = model_name
        self.stage_id = stage_id

        full = GPT2LMHeadModel.from_pretrained(model_name)
        blocks = list(full.transformer.h)
        n = len(blocks)
        per_stage = n // n_stages
        extra = n % n_stages

        self.layer_start = stage_id * per_stage + min(stage_id, extra)
        self.layer_end = self.layer_start + per_stage + (1 if stage_id < extra else 0)

        self.blocks = nn.ModuleList(blocks[self.layer_start:self.layer_end])
        self.is_first = stage_id == 0
        self.is_last = stage_id == n_stages - 1

        if self.is_first:
            self.wte = full.transformer.wte
            self.wpe = full.transformer.wpe
            self.drop = full.transformer.drop

        if self.is_last:
            self.ln_f = full.transformer.ln_f
            self.lm_head = full.lm_head

        del full

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_first:
            pos = torch.arange(x.size(1), device=x.device)
            x = self.wte(x) + self.wpe(pos)
            x = self.drop(x)

        for block in self.blocks:
            x = block(x)[0]

        if self.is_last:
            x = self.ln_f(x)
            x = self.lm_head(x[:, -1:, :])

        return x
