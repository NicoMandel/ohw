FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 
# update to 12.3 in the future!

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl ca-certificates sudo git bzip2 libx11-6 wget && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && apt install -y
#  python3.11
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y python3-pip 

RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY requirements.txt .
# insert - delete all lines starting with "torch*"
RUN sed -i '/^torch/d' requirements.txt
RUN sed -i '/^numpy/ s/./#&/' requirements.txt
# RUN pip install ultralytics
# RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install opencv-python


RUN adduser --disabled-password --gecos '' ubuntu && adduser ubuntu sudo && echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' /home/ubuntu/.bashrc
COPY test_gpu.py /home/ubuntu
USER ubuntu
WORKDIR /home/ubuntu

# ENTRYPOINT ["python3", "test_gpu.py"]