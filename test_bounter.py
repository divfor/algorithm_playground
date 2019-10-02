#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from HashBinTop import HashTop
from binascii import unhexlify
import random

def get_random_bytes(n):
    ASCII = "".join(chr(x) for x in range(32))
    s ="".join(random.choice(ASCII) for _ in range(n+3))
    return bytes(s,'ascii')[:n]

def text2hash(m,n):
    filename = '../data/netdata/web/normalTrafficTraining.txt'
    #filename = '../data/netdata/hyt/netdata.txt'
    #filename = '../data/netdata/hyt/log.pcap.1568089974.csv'
    #d = pd.read_csv(filename)
    np.random.seed(datetime.now().microsecond)
    dt = np.dtype([('counter','i4'),('n-gram',bytes,4)])
    os.system("rm -rf bn.npy")
    h = HashTop("bn.npy", 10, 10**8, m, dt)
    f = open(filename, 'rb')
    k = 0
    for line in f.readlines():
    #for ln in d['payload']:
        #line = unhexlify(ln)
        sz = len(line)
        if sz < 5: continue
        for i in range(sz-3):
            h.add(line[i:i+4])
            k += 1
            if k % 100000 == 0:
                h.summary()
        if n > 0 and k > n: break
    h.close()
    f.close()
    os.system("mv bn.npy bn.newest.npy")


def pipeline_simulated (m=10000007,n=1000):
    np.random.seed(datetime.now().microsecond)
    dt = np.dtype([('counter','i4'),('n-gram',bytes,4)])
    os.system("rm -rf bn.npy")
    h = HashTop("bn.npy", 10, 10**8, m, dt)
    for i in range(n):
        h.add(get_random_bytes(4))
    h.close()

def main():
    parser = argparse.ArgumentParser(description='simulate and compare hash k-seating issue')
    parser.add_argument("-m", "--hash_size", type=int, help="allowed max items to add")
    parser.add_argument("-n", "--uniq_size", type=int, help="allowed max items to add")
    args = parser.parse_args()
    m, n = int(args.hash_size), int(args.uniq_size)
    #pipeline_simulated(m, n)
    text2hash(m,n)

if __name__ == '__main__':
    main()
