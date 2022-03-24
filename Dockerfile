FROM nvcr.io/nvidia/pytorch:22.02-py3

WORKDIR ./YOLOX
ADD . .

RUN pip3 install -r requirements.txt

