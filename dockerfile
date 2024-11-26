FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 
# update to 12.3 in the future!

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl ca-certificates sudo git bzip2 libx11-6 wget && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && apt install -y
#  python3.11
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 tree -y
RUN apt-get install -y python3-pip 

RUN --mount=type=cache,target=~/.cache/pip python3 -m pip install torch torchvision torchaudio  --index-url https://download.pytorch.org/whl/cu118 --default-timeout=3000 
# --no-cache-dir
COPY requirements.txt .
# insert - delete all lines starting with "torch*"
RUN sed -i '/^torch/d' requirements.txt && sed -i '/^numpy/ s/./#&/' requirements.txt

# RUN pip install ultralytics
# RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install opencv-python rawpy openpyxl pyexiftool

RUN adduser --disabled-password --gecos '' ubuntu && adduser ubuntu sudo && echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' /home/ubuntu/.bashrc
COPY test_gpu.py /home/ubuntu
# these will allow the import of ohw.* in scripts, as ohw is a subdirectory of /home/ubuntu
# COPY scripts/ /home/ubuntu/
# COPY src/ohw/ /home/ubuntu/ohw/
# this may have to be changed
# COPY data/ /home/ubuntu/data/
# datasets will still have to be included as a volume
USER ubuntu
WORKDIR /home/ubuntu

# ENTRYPOINT ["python3", "test_gpu.py"]