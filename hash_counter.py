#!/usr/bin/env python3
import argparse
import numpy as np
from datetime import datetime

# n - Keys, m - Slots, k - Seats

# dt = np.dtype([('count', '<i2'), ('ngram', np.byte, 5)])
# h = np.zeros(10**8, dtype=dt)

def seating_calculated (m, n, k):
    p0 = np.exp(-k*n/m) # simplify from np.power(1-1/m, n*k)
    p1 = 1 - p0
    p_add_collision = np.power(p1, k) # when n-1 item added
    # Prob(N) fix-factor: 1/(k+1) when sum(P(i)^k), i=1,2,3,...,N
    p_add_collision_mean = p_add_collision / (1 + k)
    add_no = int(n * p_add_collision_mean)
    add_ok = n - add_no
    print("prob_calc: add_ok=(%d) add_no=(%d) p(bit-0)=(%.12f)" % (add_ok, add_no, p0))
    return add_ok

def seating_simulated (m, n, k):
    np.random.seed(datetime.now().microsecond)
    AM = np.zeros(m, dtype=np.int)
    KN = np.random.randint(0, m, size=(n, k))
    add_ok, add_no = 0, 0
    for seats in KN:
        seats_available = seats[AM[seats] == 0]
        if seats_available.size > 0:
            AM[seats_available] = 1
            add_ok += 1
        else:
            add_no += 1 # collisions + repeats
    p0 = 1.0 * len(AM[AM == 0]) / m
    print("simulated: add_ok=(%d) add_no=(%d) p(bit-0)=(%.12f)" % (add_ok, add_no, p0))
    return add_ok

def main():
    parser = argparse.ArgumentParser(description='simulate and compare hash k-seating issue')
    parser.add_argument("-k", "--hashes", type=int, help="number of Hash functions`")
    parser.add_argument("-n", "--capacity", type=float, help="allowed max items to add")
    args = parser.parse_args()
    # the best k (bits) for bloom-filter is integer round(4-drop-5-add) of ln(2)*M/N
    # M=10^9+7,K=8 ==> capacity=86643398.1765
    M = int(args.capacity * args.hashes / np.log(2))
    N, K = int(args.capacity), int(args.hashes)
    print("M N K M/N: ", M, N, K, M/N)
    pin = seating_calculated(M, N, K)
    sin = seating_simulated(M, N, K)
    print("pin/sin = %f" % (pin/sin))

if __name__ == '__main__':
    main()
