#!/bin/bash

#rm -rf ./model/*
rm -rf ./log/*

CUDA_VISIBLE_DEVICES='' python -u train.py --ps_hosts=10.141.160.46:2232 --worker_hosts=10.141.160.46:2233,10.141.160.46:2234,10.141.160.46:2235,10.141.160.46:2236 --job_name=ps --task_index=0 1>>./log/worker.log 2>&1 &
CUDA_VISIBLE_DEVICES='0' python -u train.py --ps_hosts=10.141.160.46:2232 --worker_hosts=10.141.160.46:2233,10.141.160.46:2234,10.141.160.46:2235,10.141.160.46:2236 --job_name=worker --task_index=0 1>>./log/worker.log 2>&1 &
CUDA_VISIBLE_DEVICES='1' python -u train.py --ps_hosts=10.141.160.46:2232 --worker_hosts=10.141.160.46:2233,10.141.160.46:2234,10.141.160.46:2235,10.141.160.46:2236 --job_name=worker --task_index=1 1>>./log/worker.log 2>&1 &
CUDA_VISIBLE_DEVICES='2' python -u train.py --ps_hosts=10.141.160.46:2232 --worker_hosts=10.141.160.46:2233,10.141.160.46:2234,10.141.160.46:2235,10.141.160.46:2236 --job_name=worker --task_index=2 1>>./log/worker.log 2>&1 &
CUDA_VISIBLE_DEVICES='3' python -u train.py --ps_hosts=10.141.160.46:2232 --worker_hosts=10.141.160.46:2233,10.141.160.46:2234,10.141.160.46:2235,10.141.160.46:2236 --job_name=worker --task_index=3 1>>./log/worker.log 2>&1 &


#CUDA_VISIBLE_DEVICES='' mypython -u train.py --ps_hosts=10.141.160.46:2222 --worker_hosts=10.141.160.46:2223 --job_name=ps --task_index=0 &
#CUDA_VISIBLE_DEVICES='4' mypython -u train.py --ps_hosts=10.141.160.46:2222 --worker_hosts=10.141.160.46:2223 --job_name=worker --task_index=0

#CUDA_VISIBLE_DEVICES='' mypython predict.py
