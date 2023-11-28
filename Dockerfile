FROM nvcr.io/nvidia/pytorch:21.06-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt update && apt install -y git vim tmux htop python3-pip software-properties-common libgl1 wget

WORKDIR /root

RUN git clone https://github.com/open-mmlab/mmcv.git

WORKDIR /root/mmcv

RUN git checkout v1.3.9

ARG TORCH_CUDA_ARCH_LIST=8.6

RUN MMCV_WITH_OPS=1 pip install -e .

WORKDIR /root

RUN git clone https://github.com/yanghoonkim/ViTPose.git

WORKDIR /root/ViTPose

RUN git checkout nia

RUN pip install -v -e .

RUN pip install timm==0.4.9 einops numpy==1.23.5 opencv-python==4.5.5.64 yapf==0.40.1 future tensorboard pandas notebook==6.4.8 traitlets==5.9.0 polars

CMD bash