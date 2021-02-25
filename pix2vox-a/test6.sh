#!/bin/bash
mkdir -p spatial_dag/weight_1/epsilon_1
python runner.py --alpha_inp=5e-2 --alpha_flow=1e-4 --attack_epsilon=1 --attack_type=spatial_dag --weight=1 --source tab case mail --target chair desk car tab case mail > spatial_dag/weight_1/epsilon_1/out_tab_case_mail.txt
mkdir -p spatial_dag/weight_1/epsilon_2
python runner.py --alpha_inp=1e-1 --alpha_flow=1e-4 --attack_epsilon=2 --attack_type=spatial_dag --weight=1 --source tab case mail --target chair desk car tab case mail > spatial_dag/weight_1/epsilon_2/out_tab_case_mail.txt
mkdir -p spatial_dag/weight_1/epsilon_4
python runner.py --alpha_inp=5e-1 --alpha_flow=2e-4 --attack_epsilon=4 --attack_type=spatial_dag --weight=1 --source tab case mail --target chair desk car tab case mail > spatial_dag/weight_1/epsilon_4/out_tab_case_mail.txt
mkdir -p spatial_dag/weight_1/epsilon_16
python runner.py --alpha_inp=1 --alpha_flow=2e-4 --attack_epsilon=16 --attack_type=spatial_dag --weight=1 --source tab case mail --target chair desk car tab case mail > spatial_dag/weight_1/epsilon_16/out_tab_case_mail.txt
