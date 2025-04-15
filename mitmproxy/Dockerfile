FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install mitmproxy

COPY mitm_addon.py /addon/mitm_addon.py

WORKDIR /mitmproxy
CMD ["mitmdump", "-p", "8080", "--mode", "reverse:https://api.openai.com", "-s", "/addon/mitm_addon.py"]
