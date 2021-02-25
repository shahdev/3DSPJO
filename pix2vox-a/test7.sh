#!/bin/bash
mkdir -p dag/weight_1/epsilon_1
python runner.py --alpha_inp=5e-2 --alpha_flow=0 --attack_epsilon=1 --attack_type=dag --weight=1 --source chair desk car --target chair desk car tab case mail > dag/weight_1/epsilon_1/out_chair_desk_car.txt
mkdir -p dag/weight_1/epsilon_2
python runner.py --alpha_inp=1e-1 --alpha_flow=0 --attack_epsilon=2 --attack_type=dag --weight=1 --source chair desk car --target chair desk car tab case mail > dag/weight_1/epsilon_2/out_chair_desk_car.txt
mkdir -p dag/weight_1/epsilon_4
python runner.py --alpha_inp=2e-1 --alpha_flow=0 --attack_epsilon=4 --attack_type=dag --weight=1 --source chair desk car --target chair desk car tab case mail > dag/weight_1/epsilon_4/out_chair_desk_car.txt
mkdir -p dag/weight_1/epsilon_16
python runner.py --alpha_inp=5e-1 --alpha_flow=0 --attack_epsilon=16 --attack_type=dag --weight=1 --source chair desk car --target chair desk car tab case mail > dag/weight_1/epsilon_16/out_chair_desk_car.txt
