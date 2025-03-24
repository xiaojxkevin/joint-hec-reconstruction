#!/bin/bash
# sinfo -O Nodehost,Gres:.30,GresUsed:.45

# salloc -N 1 -n 10 --gres=gpu:NVIDIATITANV:1
salloc -N 1 -n 10 --gres=gpu:NVIDIAGeForceRTX2080Ti:1

