import argparse
import logging
import time

import grpc
import torch
from transformers import AutoTokenizer

from src.generated import pipeline_pb2, pipeline_pb2_grpc
from src.serialize import encode, decode

log = logging.getLogger(__name__)

MAX_MSG = 256 * 1024 * 1024


class Pipeline:

    def __init__(self, worker_addrs: list[str], model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.stubs = []
        for addr in worker_addrs:
            ch = grpc.insecure_channel(
                addr,
                options=[
                    ("grpc.max_send_message_length", MAX_MSG),
                    ("grpc.max_receive_message_length", MAX_MSG),
                ],
            )
            self.stubs.append(pipeline_pb2_grpc.WorkerStub(ch))

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 1.0) -> str:
        ids = self.tokenizer.encode(prompt, return_tensors="pt")

        for _ in range(max_tokens):
            data, shape, dtype = encode(ids)
            msg = pipeline_pb2.Tensor(data=data, shape=shape, dtype=dtype)

            for stub in self.stubs:
                msg = stub.Forward(msg)

            logits = decode(msg.data, list(msg.shape), msg.dtype)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            ids = torch.cat([ids, next_id], dim=1)

            token = self.tokenizer.decode(next_id[0])
            print(token, end="", flush=True)

            if next_id.item() == self.tokenizer.eos_token_id:
                break

        print()
        return self.tokenizer.decode(ids[0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--workers", default="localhost:50051,localhost:50052")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    addrs = args.workers.split(",")
    pipe = Pipeline(addrs, args.model)

    t0 = time.perf_counter()
    result = pipe.generate(args.prompt, args.max_tokens, args.temperature)
    elapsed = time.perf_counter() - t0

    log.info(f"{len(result.split())} tokens in {elapsed:.1f}s")
