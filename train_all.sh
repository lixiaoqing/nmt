#!/bin/bash

export THEANO_FLAGS=device=gpu2,floatX=float32
python ./train_nmt_all.py



