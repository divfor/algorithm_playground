#!/usr/bin/env python3
import os
import argparse
import numpy as np
from datetime import datetime
from HashBinTop import HashTop

def pipeline_simulated (m=10000007,n=1000):
    np.random.seed(datetime.now().microsecond)
    dt = np.dtype([('counter','u2'),('key',np.int)])
    real = 100000
    os.system("rm -rf bn.npy")
    h = HashTop("bn.npy", 2, 65530, m, dt)
    for i in np.random.randint(0, real, n):
        h.add(i)
    h.close()
    nb = h.ht
    for d in nb:
        if abs(d[0]) >=65530:
            print(d)

def main():
    parser = argparse.ArgumentParser(description='simulate and compare hash k-seating issue')
    parser.add_argument("-m", "--hash_size", type=int, help="allowed max items to add")
    parser.add_argument("-n", "--uniq_size", type=int, help="allowed max items to add")
    args = parser.parse_args()
    m, n = int(args.hash_size), int(args.uniq_size)
    pipeline_simulated(m, n)

if __name__ == '__main__':
    main()
