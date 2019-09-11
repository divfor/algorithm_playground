#!/usr/bin/env python3

# https://github.com/escherba/python-cityhash
# https://github.com/google/cityhash
# https://github.com/ekzhu/datasketch
# https://github.com/RaRe-Technologies/bounter
# from datasketch import HyperLogLogPlusPlus
from cityhash import CityHash64WithSeed as cityhash
from bounter import bounter
import numpy as np

class HashTop(object):
    '''
        dumpfile : 'path/to/filename.npy' to be loaded by numpy
        lowfreq_threshold : preemptable threshold under which bucket could be overwritten by new key
        highfreq_threshold : ceiling of counters to limit increasing or avoid overflow
        hash_size : hash table size
        hash_dtype : example - numpy.dtype([('counter', 'u2'), ('n-gram', bytes, 5)])
    '''
    def __init__(self, dumpfile = None, 
                lowfreq_threshold = 0, highfreq_threshold = 65535, hash_size = 7+10**8,
                hash_dtype = np.dtype([('counter', 'u2'), ('n-gram', bytes, 5)])):
        self.hash_dtype = hash_dtype
        self.hash_size = hash_size
        self.hash_dumpfile = dumpfile
        self.lowfreq_threshold = lowfreq_threshold
        self.highfreq_threshold = highfreq_threshold
        self.hash_add_tries = 0 # sum of hasd-add calls
        self.hash_added_keys = 0 # num of buckets in use
        self.hash_relookups = 0 # sum of re-lookups of all hash-add calls
        self.hash_collisions = 0 # sum of hash-add calls which fail on all re-lookups
        self.hash_ceilings = 0 # sum of hash-add calls which counter overflows highfreq_threshold
        self.hash_overwrites = 0 # num of hash-add calls which key overwrites another one and counter resets to 1
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
        print("CityHash relookups:   %ld" % self.hash_relookups)
        print("CityHash collisions:  %ld" % self.hash_collisions)
        print("CityHash ceilings:    %ld" % self.hash_ceilings)
        print("CityHash overwrites:  %ld" % self.hash_overwrites)
        print("CityHash added_keys:  %ld" % self.hash_added_keys)
        print("Bounter HyperLogLog:  %ld\n" % self.bnt.cardinality())

    def add(self, ngram): # bytes type
        self.hash_add_tries += 1
        self.bnt.update([bytes(ngram)])
        n_left_hash_funcs = self.hash_funcs_num
        for seed in self.hash_seeds:
            hv = cityhash(bytes(ngram), seed) % self.hash_size
            i = hv % self.hash_size
            step = 1 if 1 == (hv % 2) else -1
            absc = abs(self.ht[i][0])
            n_left_hash_funcs -= 1
            # hash bucket is not in use
            if self.ht[i][1] == 0:
                self.hash_added_keys += 1
                self.ht[i] = (step, ngram)
                break
            # hash bucket is owned already
            if self.ht[i][1] == ngram:
                if (absc < self.highfreq_threshold):
                    self.ht[i][0] += step
                else:
                    self.hash_ceilings += 1
                break
            # hash bucket is owned by others, try other buckets
            if absc > self.lowfreq_threshold:
                if n_left_hash_funcs == 0:
                    self.hash_collisions += 1 # tried all buckets and give up here
                    break
                self.hash_relookups += 1
                continue
            # hash bucket is under low-freq-protection
            self.ht[i][0] += step
            if self.ht[i][0] == 0:
                self.hash_overwrites += 1 # kick out low-freq ngram
                self.ht[i][1] = ngram
            break

