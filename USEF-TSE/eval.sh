#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python eval.py  --config chkpt/USEF-TFGridNet/config.yaml \
                --chkpt-path chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar \
                --test-set wsj0-2mix #['wsj0-2mix', 'wham!', 'whamr!']

# python eval.py  --config chkpt/USEF-SepFormer/config.yaml \
#                 --chkpt-path chkpt/USEF-SepFormer/wsj0-2mix/temp_best.pth.tar \
#                 --test-set wsj0-2mix #['wsj0-2mix', 'wham!', 'whamr!']