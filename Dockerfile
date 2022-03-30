FROM nvcr.io/nvidia/pytorch:22.02-py3

WORKDIR ./YOLOX
ADD . .

ENV TZ=US/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
apt-get update && \ 
apt-get upgrade -y && \
apt-get install ffmpeg libsm6 libxext6  -y && \
git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
cd torch2trt && \
python3 setup.py install --plugins && \
cd .. && \
pip3 install -U pip && pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
pip3 install -v -e .


