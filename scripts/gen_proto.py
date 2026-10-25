import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "src" / "generated"
out.mkdir(parents=True, exist_ok=True)
(out / "__init__.py").touch()

subprocess.run(
    [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={root / 'proto'}",
        f"--python_out={out}",
        f"--grpc_python_out={out}",
        str(root / "proto" / "pipeline.proto"),
    ],
    check=True,
)

grpc_file = out / "pipeline_pb2_grpc.py"
text = grpc_file.read_text()
text = text.replace(
    "import pipeline_pb2 as pipeline__pb2",
    "from . import pipeline_pb2 as pipeline__pb2",
)
grpc_file.write_text(text)

print("proto stubs generated")
