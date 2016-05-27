#!/bin/bash

export THEANO_FLAGS=device=gpu,floatX=float32

python ./translate.py -n -k 10 \
	model.npz  \
	/home/xqli/data/big/big.ch.pkl \
	/home/xqli/data/big/big.en.pkl \
	/home/xqli/data/nist/03.seg \
    03.out
