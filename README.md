# RF-Tracker

Rotated-Fisheye Tracker is a simple motion tracker that transforms rotated bounding box detections into 2d Gaussians distributions and then uses distribution distances to associate which replicates the IoU association behaviour.

## Abstract

## Installation
### 1. Installing on the host machine
Step1. Install RF-Tracker.
```shell
git clone https://github.com/Skyteens/RF-Tracker.git
cd RF-Tracker
conda create -n <ENV_NAME> python=3.8
conda activate 
pip install -r requirements.txt

```


## Demo
Run RF-Tracker:

1. prepare a folder of each video frame and its rotated bounding box detections

```shell
cd <RF-Tracker_HOME>
python3 demo.py
```

2. For different association methods
```shell
python3 demo.py --match [gwd,bd,kld]
```

## Acknowledgement

A large part of the code is borrowed from [ByteTrack](https://github.com/ifzhang/ByteTrack), [MMRotate](https://github.com/open-mmlab/mmrotate), and [RAPiD](https://github.com/duanzhiihao/RAPiD). Many thanks for their wonderful works.
