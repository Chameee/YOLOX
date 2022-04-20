# Video Object Detection Using YOLOX

## Introduction

Detecting person and objects in live video can be concluded into a subtask of **Object Dection**. In order to apply deep learning model on large scale live-streaming video, we develop a detection image easy to deploy and inference & export results. 

Details of *YOLOX* architecture and intro can be found [**Here**](./README_ori.md)

For more information about *Nvidia Container*, see [**here**](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

For common *Docker* commands, see [**here**](https://yeasy.gitbook.io/docker_practice/image/dockerfile)

## QuickStart

### Pre-install

Make sure you have installed [**Docker**](https://docs.docker.com/engine/install/ubuntu/)


### Download and Build Docker image

```
git clone git@github.com:Chameee/YOLOX.git
cd YOLOX
docker build . -t yolox_deploy:v1
```

### Maintain docker container.

Replace *{YOUR_DATA_SAVEDIR}* using your actual data save dir, **e.g.** */home/Andy/data/* 

- Run container once (all data generated in container will be deleted after close terminal)

```
docker run --gpus all -v {YOUR_DATA_SAVEDIR}:/data -it yolox_deploy:v1
```

- Run container in backend (container will remain alive after close termial)
```
docker run --gpus all -v {YOUR_DATA_SAVEDIR}:/data -itd yolox_deploy:v1
```

**Hint:** `docker ps` may help you check which container is alive.

### Run Inference

First convert *PyTorch* exported checkpoint into *TensorRT* to accelerate interence speed (10x ~ 100x).

[YOLOX-S Checkpoint Download link](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)

```
python3 tools/trt.py -n yolox-s  -c /data/yolox_s.pth
```

Then run a demo video and see result! (under *./YOLOX_outputs/yolox_s/vis_res/{timestamp}/detect_result.json*)

[Demo video link](https://drive.google.com/file/d/1YJafoCLtB6vNlyORaXyZOGnYJllSZVlF/view?usp=sharing)

```
python3 tools/demo.py video -n yolox-s --trt -c YOLOX_outputs/yolox_s/model_trt.pth --path /data/3min_with_bgm.mp4 --frame_sample_interval 2.0 --save_result_json
```

## Custom Args

`--frame_sample_interval`: Set how long seconds (*e.g.* 0.1) to capture a frame from video, default capture all frames. 

`--save_result_json`: Set whether detection result is saved in json format. 



