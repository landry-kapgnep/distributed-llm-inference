.PHONY: proto install test clean

proto:
	python scripts/gen_proto.py

install:
	pip install -e ".[dev]"

worker-0:
	python -m src.worker --stage 0 --stages 2 --port 50051

worker-1:
	python -m src.worker --stage 1 --stages 2 --port 50052

run:
	python -m src.coordinator "The meaning of life is" --workers localhost:50051,localhost:50052

test:
	python -m pytest tests/ -v

clean:
	python -c "import shutil; shutil.rmtree('src/generated', True)"
