#!/bin/bash

echo "===================================================="
echo " Running all CuLane experiments "
echo "===================================================="

echo "===================================================="
echo " TRAINING: ResNet18 on CULane "
echo "===================================================="

python main.py train --exp_name culane_resnet18 --cfg cfgs/laneatt_culane_resnet18.yml

echo "===================================================="
echo " FINISHED: ResNet18 "
echo "===================================================="
echo ""


echo "===================================================="
echo " TRAINING: ResNet34 on CULane "
echo "===================================================="

python main.py train --exp_name culane_resnet34 --cfg cfgs/laneatt_culane_resnet34.yml

echo "===================================================="
echo " FINISHED: ResNet34 "
echo "===================================================="
echo ""


echo "===================================================="
echo " TRAINING: ResNet122 on CULane "
echo "===================================================="

python main.py train --exp_name culane_resnet122 --cfg cfgs/laneatt_culane_resnet122.yml

echo "===================================================="
echo " FINISHED: ResNet122 "
echo "===================================================="
echo ""

echo "===================================================="
echo " ALL TRAINING COMPLETE "
echo "===================================================="
