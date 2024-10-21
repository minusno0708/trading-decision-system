# Cuda11.7
FROM nvidia/cuda:11.7.1-base-ubuntu22.04

WORKDIR /workspace

COPY . .

RUN apt update && apt install -y python3.11 python3-pip

CMD ["nvidia-smi"]