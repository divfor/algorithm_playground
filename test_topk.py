#!/usr/bin/env python3
import os
import argparse
import numpy as np
from datetime import datetime
from HashTop import HashTop

def pipeline_simulated (m=10000007,n=1000):
    np.random.seed(datetime.now().microsecond)
    dt = np.dtype([('counter',np.int),('key',np.int)])
    real = 10000
    j, threshold = 0, 95*real/100
    os.system("rm -rf bn.npy")
    h = HashTop("bn.npy", 1, 2**32-1, m, dt)
    for i in np.random.randint(-1000, 1000, n):
        h.add(i)
        j += 1
        if h.bnt.cardinality() > threshold:
            print("hit estimated when step is %ld" % j)
            break
    h.close()

def main():
    parser = argparse.ArgumentParser(description='simulate and compare hash k-seating issue')
    parser.add_argument("-m", "--hash_size", type=int, help="allowed max items to add")
    parser.add_argument("-n", "--uniq_size", type=int, help="allowed max items to add")
    args = parser.parse_args()
    m, n = int(args.hash_size), int(args.uniq_size)
    pipeline_simulated(m, n)

if __name__ == '__main__':
    main()
