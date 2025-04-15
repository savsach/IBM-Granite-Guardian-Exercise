FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

RUN pip install transformers fastapi uvicorn accelerate

WORKDIR /app
COPY guardian.py .

CMD ["uvicorn", "guardian:app", "--host", "0.0.0.0", "--port", "8000"]