#!/usr/bin/env python3

# https://github.com/escherba/python-cityhash
# https://github.com/google/cityhash
# https://github.com/ekzhu/datasketch
# https://github.com/RaRe-Technologies/bounter
# from datasketch import HyperLogLogPlusPlus
from cityhash import CityHash64WithSeed as cityhash
from bounter import bounter
import numpy as np
import pandas as pd

class HashTop(object):
    '''
        dumpfile : 'path/to/filename.npy' to be loaded by numpy
        lowfreq_threshold : preemptable threshold under which bucket could be overwritten by new key
        highfreq_threshold : ceiling of counters to limit increasing or avoid overflow
        hash_size : hash table size
        hash_dtype : example - numpy.dtype([('counter', 'u2'), ('n-gram', bytes, 5)])
    '''
    def __init__(self, dumpfile = None, 
                lowfreq_threshold = 10, highfreq_threshold = 10**8, hash_size = 7+10**8,
                hash_dtype = np.dtype([('counter', 'i4'), ('n-gram', bytes, 4)])):
        self.hash_dtype = hash_dtype
        self.hash_size = hash_size
        self.hash_dumpfile = dumpfile
        self.lowfreq = lowfreq_threshold
        self.door = lowfreq_threshold
        self.highfreq = highfreq_threshold
        self.hash_add_tries = 0 # sum of hasd-add calls
        self.hash_added_keys = 0 # num of buckets in use
        self.hash_relookups = 0 # sum of re-lookups of all hash-add calls
        self.hash_collisions = 0 # sum of hash-add calls which fail on all re-lookups
        self.hash_ceilings = 0 # sum of hash-add calls which counter overflows highfreq_threshold
        self.hash_overwrites = 0 # num of hash-add calls which key overwrites another one and counter resets to 1
        self.hash_added_tries = 0
        self.p = 1.0/self.lowfreq #1 - np.power(0.5, 1.0/self.lowfreq)
        self.hash_counter_lost = 0 # sum of counters when key is overwritten
        self.bnt = bounter(need_counts=False) # use HLL algorithm only
        self.bnt_count = 0
        self.ht = None
        self.hash_seeds = [2819948058, 5686873063, 1769651746, 8745608315, 2414950003, 
        3583714723, 1224464945, 2514535028] #np.random.randint(10**9,10**10,8)
        self.hash_funcs_num = len(self.hash_seeds)

        if dumpfile:
            self.ht = self.load(dumpfile)
        if self.ht is None:
            self.ht = self.init()
        if self.ht is not None:
            print("hash_table dtype(%s)" % self.ht.dtype)
            print("hash_size(%d): %s" % (len(self.ht), self.ht))
        else:
            print("hash_table load() or init() failed.")

    def init(self):
        self.ht = np.zeros(self.hash_size, dtype=self.hash_dtype)
        return self.ht

    def load(self, dumpfile):
        try:
            self.ht = np.load(dumpfile)
        except:
            pass
        if self.ht is not None:
            self.hash_dtype = self.ht.dtype
            self.hash_size = len(self.ht)
        return self.ht

    def save(self, dumpfile=None):
        npyfile = dumpfile or self.hash_dumpfile
        print("\nSaving hash to %s" % npyfile)
        np.save(npyfile, self.ht)

    def close(self):
        self.save()
        self.summary()
    
    def summary(self):
        k, n, m = self.hash_funcs_num, self.bnt.cardinality(), self.hash_size
        b, colname = self.ht, self.ht.dtype.names[1]
        # CollisionOut_Prob(N) fix-factor: 1/(k+1) when sum(P(i)^k), i=1,2,3,...,N
        p0 = np.power(n/m, k)/(k+1)
        # p0/p == (N/m)^k/(n/m)^k = (1+p)^k => p ~= (sqrt(1+4k*p0)-1)/(2k)
        p = (np.sqrt(1 + 4*k*p0) - 1)/(2*k)
        p = min(max(0.0,p),1.0)
        noSeat = int(n * p)
        e = n - noSeat
        r = self.hash_added_keys # m - len(b[b[colname] == b''])
        #t = self.hash_overwrites / self.hash_added_keys
        all = abs(b[b['n-gram'] != b'']['counter'])
        mean, std = np.mean(all), np.std(all)
        part = all[all < self.door]
        pmean, pstd, pcent = np.mean(part), np.std(part), len(part)/len(all)
        #cmean, cstd = np.sqrt(t*(2/3.145926)), np.sqrt(t*(1-2/3.1415926))
        print("lowfreq: %ld, highfreq: %ld, num_hash_funcs: %d" % (self.lowfreq, self.highfreq, self.hash_funcs_num))
        print("CityHash collide rate:%.12f %%" % (100.0*p))
        print("CityHash collisions:  %ld (%ld estimated)" % (self.hash_collisions, noSeat))
        print("CityHash ceilings:    %ld" % self.hash_ceilings)
        print("CityHash relookups:   %ld" % self.hash_relookups)
        print("CityHash overwrites:  %ld" % self.hash_overwrites)
        print("CityHash added_tries: %ld" % self.hash_added_tries)
        print("CityHash add_tries:   %ld" % self.hash_add_tries)
        print("CityHash added_keys:  %ld" % self.hash_added_keys)
        print("Bounter HyperLogLog:  %ld" % n)
        print("Estimated loadfactor: %ld / %ld = %.6f" % (e, m, e/m))
        print("Actual loadfactor:    %ld / %ld = %.6f" % (r, m, r/m))
        print("CityHash all counters >> std: %.2f, mean/door: %.2f/%ld, p_overwrite: %.4f" % (std, mean, self.door, self.p))
        print("CityHash overwritable >> std: %.2f, mean/door: %.2f/%ld, ow_percent: %.4f\n" % (pstd, pmean, self.door, pcent))

    def get(self, ngram): # bytes type
        for seed in self.hash_seeds:
            i = cityhash(bytes(ngram), seed) % self.hash_size
            if self.ht[i][1] == ngram:
                return abs(self.ht[i][0])
        return 0

    def add(self, ngram, count=1): # bytes type
        self.hash_add_tries += count
        self.bnt.update([bytes(ngram)])
        self.door = max(2, self.hash_added_tries // (1 + self.hash_added_keys))
        n_left_hash_funcs = self.hash_funcs_num
        i_ow, n_ow = -1, -1 # remember lowest bucket for overwritting
        for seed in self.hash_seeds:
            hv = cityhash(bytes(ngram), seed)
            i = hv % self.hash_size
            sign = ((hv >> 32) ^ hv) % 2
            step = count if 1 == sign else -count
            absc = abs(self.ht[i][0])
            n_left_hash_funcs -= 1
            # bucket is empty:
            if self.ht[i][1] == b'':
                self.hash_added_keys += 1
                self.hash_added_tries += count
                self.ht[i] = (step, ngram)
                break
            # bucket is owned:
            if self.ht[i][1] == ngram:
                if (absc < self.highfreq):
                    self.ht[i][0] += step
                    self.hash_added_tries += count
                else:
                    self.hash_ceilings += 1
                break
            # bucket is owned by others:
            if absc < self.door:
                if i_ow < 0 or absc < n_ow:
                    i_ow, n_ow, step_ow = i, absc, step
            if n_left_hash_funcs > 0:
                continue
            # n_left_hash_funcs == 0:
            if i_ow < 0:
                self.hash_collisions += 1
                break
            self.hash_relookups += 1
            self.ht[i_ow][0] += step_ow
            self.hash_added_tries += count
            if np.random.random() < self.p:
                self.hash_overwrites += 1
                self.ht[i_ow][1] = ngram
            break

    def update(self, hash_tables):
        for table in hash_tables:
            h = table.ht
            ht = h[h[1] != b'']
            for i in ht:
                self.add(i[1], abs(i[0])
