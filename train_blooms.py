#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os
from BytesNgram import BytesNgram

def cmdget(cmd):
    return [f.strip() for f in os.popen(cmd).readlines()]
    
def main():
    train_txt_num = 123
    bn = BytesNgram()
    os.system('mkdir -p ./logs && rm -rf ./logs/*')
    os.system('mkdir -p /data/bloomfilter/tmp && rm -rf /data/bloomfilter/tmp/*')

    datapath = '/data/data1/payloads/gold'
    os.system('mkdir -p %s/used' % datapath)
    for file in cmdget('ls %s/*.txt | sort' % datapath)[:train_txt_num]:
        with open(file) as f:
            bn.train_gold_bloom_filters(f.readlines())
        os.system('sudo mv -f %s %s/used/' % (file, datapath))

    datapath = '/data/data2/payloads/bad'
    os.system('mkdir -p %s/used' % datapath)
    for file in cmdget('ls %s/*.txt | sort' % datapath)[:train_txt_num]:
        with open(file) as f:
            bn.train_bad_bloom_filters(f.readlines())
        os.system('sudo mv -f %s %s/used/' % (file, datapath))

    datapath = '/data/data3/payloads/normal'
    os.system('mkdir -p %s/used' % datapath)
    for file in cmdget('ls %s/*.txt | sort' % datapath)[:train_txt_num]:
        with open(file) as f:
            bn.train_normal_bloom_filters(f.readlines())
        os.system('sudo mv -f %s %s/used/' % (file, datapath))

if __name__ == "__main__":
    main()
