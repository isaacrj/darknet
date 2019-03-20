#!/bin/sh
./darknet detector train karting_001/training.data karting_001/yolov3-tiny-obj-karting-training.cfg "$@"
