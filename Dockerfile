FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY proto/ proto/
COPY scripts/ scripts/
COPY src/ src/
RUN python scripts/gen_proto.py

ENTRYPOINT ["python", "-m", "src.worker"]
