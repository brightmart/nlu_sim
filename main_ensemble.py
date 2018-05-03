#/usr/bin/env python
#coding=utf-8
import jieba
import sys
from a1_dual_bilstm_cnn_predict_ensemble import predict_bilstm
def process(inpath, outpath):
    predict_bilstm(inpath,outpath)

if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
