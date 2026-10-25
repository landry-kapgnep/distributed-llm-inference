import argparse
import logging
from concurrent import futures

import grpc

from src.generated import pipeline_pb2, pipeline_pb2_grpc
from src.serialize import encode, decode
from src.stage import GPT2Stage

log = logging.getLogger(__name__)

MAX_MSG = 256 * 1024 * 1024


class WorkerServicer(pipeline_pb2_grpc.WorkerServicer):

    def __init__(self, model_name: str, stage_id: int, n_stages: int):
        self.model = GPT2Stage(model_name, stage_id, n_stages)
        self.stage_id = stage_id
        log.info(
            f"stage {stage_id} ready — layers [{self.model.layer_start}, {self.model.layer_end})"
        )

    def Forward(self, request, context):
        x = decode(request.data, list(request.shape), request.dtype)
        if self.model.is_first:
            x = x.long()
        out = self.model(x)
        data, shape, dtype = encode(out)
        return pipeline_pb2.Tensor(data=data, shape=shape, dtype=dtype)

    def Ping(self, request, context):
        return pipeline_pb2.WorkerInfo(
            stage=self.stage_id,
            layer_start=self.model.layer_start,
            layer_end=self.model.layer_end,
            model=self.model.model_name,
        )


def serve(port: int, model_name: str, stage_id: int, n_stages: int):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", MAX_MSG),
            ("grpc.max_receive_message_length", MAX_MSG),
        ],
    )
    pipeline_pb2_grpc.add_WorkerServicer_to_server(
        WorkerServicer(model_name, stage_id, n_stages), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info(f"listening on :{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--stages", type=int, default=2)
    args = p.parse_args()
    serve(args.port, args.model, args.stage, args.stages)
