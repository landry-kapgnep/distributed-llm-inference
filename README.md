# distributed-llm-inference

I built this to understand how pipeline parallelism works under the surface. The idea is to take a language model, split it across multiple machines and run inference over gRPC. Each worker contains a part of the transformer layers and a coordinator sends tokens through the pipeline.

This is basically the same idea as [Petals](https://github.com/bigscience-workshop/petals) but down to the core.

## How it works

A transformer (GPT-2 for now) is basically a pile of identical layers. This project cuts that pile in two and puts each half in a gRPC server:

```
tokens -> Worker 0 (embed + layers 0-5) -> Worker 1 (layers 6-11 + lm_head) -> logits
```

The coordinator tokenizes the prompt and sends the IDs to Worker 0. Worker 0 runs the embedding and the first six transformer blocks, then sends the hidden state over gRPC to Worker 1. Worker 1 finishes the forward pass and returns the logits. The coordinator samples the next token, adds it to the sequence, and loops.

## Setup

Python 3.10+, PyTorch, transformers, grpcio.

```bash
pip install grpcio grpcio-tools torch transformers numpy
```

Generate gRPC stubs:

```bash
python scripts/gen_proto.py
```

## Running it

Start both workers in separate terminals:

```bash
# terminal 1
python -m src.worker --stage 0 --stages 2 --port 50051

# terminal 2
python -m src.worker --stage 1 --stages 2 --port 50052
```

Once they both have print `listening`, run the coordinator:

```bash
python -m src.coordinator "The meaning of life is" --workers localhost:50051,localhost:50052
```

Tokens stream to stdout while they are generated.

## Docker

```bash
docker compose up --build
```

Then point the coordinator to the containers:

```bash
python -m src.coordinator "Once upon a time" --workers localhost:50051,localhost:50052
```

## Tests

Equivalence test checks that the split model outputs the same logits as the monolithic one, if this breaks nothing else matters.

```bash
python -m pytest tests/ -v
```

Both a 2-way and a 3-way split are tested

## Project layout

```
proto/pipeline.proto    gRPC service contract (Tensor message and Forward RPC)
src/stage.py            GPT2Stage nn.Module wrapping N transformer layers
src/serialize.py        tensor - bytes for gRPC transport
src/worker.py           gRPC server hosting one pipeline stage
src/coordinator.py      autoregressive generation over the workers
tests/                  equivalence tests (split output == monolithic output)
```

## Notes

GPT-2 has 12 transformer blocks that all have the same input/output shape, which is what permits to split the model at any layer limit. The tensor travelling between workers is around 300 KB for a 100-token sequence, serialized as raw bytes over gRPC (which is way faster than JSON).

There is no KV cache so each step processes again the full sequence from scratch. It's the first thing  i would optimize but for now it keeps the code simpler to read.
