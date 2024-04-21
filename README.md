# RF-Tracker

Rotated-Fisheye Tracker is a simple motion tracker that transforms rotated bounding box detections into 2d Gaussians distributions and then uses distribution distances to associate which replicates the IoU association behaviour.

## Abstract

## Installation
### 1. Installing on the host machine
Step1. Install RF-Tracker.
```shell
git clone https://github.com/Skyteens/RF-Tracker.git
cd RF-Tracker
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip3 install cython_bbox
```

## Demo
Run RF-Tracker:

1. prepare a folder of each video frame and its rotated bounding box detections
2. Run the demo.py with updated parameters

```shell
cd <RF-Tracker_HOME>
python3 demo.py
```

## Acknowledgement

A large part of the code is borrowed from [ByteTrack](https://github.com/ifzhang/ByteTrack), [MMRotate](https://github.com/open-mmlab/mmrotate), and [RAPiD](https://github.com/duanzhiihao/RAPiD). Many thanks for their wonderful works.
