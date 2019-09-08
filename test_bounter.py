#!/usr/bin/env python3
import argparse
import numpy as np
from datetime import datetime
from HashTop import HashTop

def pipeline_simulated (m=10000007,n=1000):
    np.random.seed(datetime.now().microsecond)
    dt = np.dtype([('counter','u2'),('key',np.int)])
    h = HashTop("test_hll.npy", lowfreq_threshold=1, highfreq_threshold=65535, hash_size=m, hash_dtype=dt)
    for i in np.random.randint(0, 1000000, n):
        h.add(i)
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
