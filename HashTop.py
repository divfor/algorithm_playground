#!/usr/bin/env python3
# pip3 install cityhash, see https://github.com/escherba/python-cityhash
# https://github.com/google/cityhash
from cityhash import CityHash64WithSeed as cityhash
import numpy as np

class HashTop(object):
    def __init__(self,
    dumpfile = None, # np.load('x.npy') if file exists, instead of init()
    lowfreq_threshold = 0, # under which bucket is overwritable
    highfreq_threshold = 65535, # counter's ceiling
    hash_size = 7+10**8, # hash table size
    hash_dtype = np.dtype([('counter', 'u2'), ('n-gram', bytes, 5)])
    ):
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
        self.hash_seeds = [2819948058, 5686873063, 1769651746, 8745608315, 2414950003, 
        3583714723, 1224464945, 2514535028] #np.random.randint(10**9,10**10,8)
 
        if dumpfile:
            self.ht = self.load(dumpfile)
        if self.ht is None:
            self.ht = self.init()
        if self.ht:
            print("hash_table dtype(%s)\nhash_size(%d): %s" % (self.ht.dtype, len(self.ht), self.ht))
        else:
            print("hash_table load() or init() failed.")

    def init(self):
        self.ht = np.zeors(self.hash_size, dtype=self.hash_dtype)
        return self.ht

    def load(self, dumpfile):
        try:
            self.ht = np.load(dumpfile)
        except:
            pass
        if self.ht:
            self.hash_dtype = self.ht.dtype
            self.hash_size = len(self.ht)
        return self.ht

    def save(self, dumpfile=None):
        npyfile = dumpfile or self.hash_dumpfile
        print("Saving hash to %s" % npyfile)
        np.save(npyfile, self.ht)

    def add(self, ngram): # bytes type
        self.hash_add_tries += 1
        n_left_hash_funcs = self.hash_funcs_num
        for seed in self.hash_seeds:
            i = cityhash(ngram.hex(), seed) % self.hash_size
            n_left_hash_funcs -= 1
            if self.ht[i][1] == ngram:
                if (self.ht[i][0] < self.highfreq_threshold):
                    self.ht[i][0] += 1
                else:
                    self.hash_ceilings += 1
            else:
                if self.ht[i][0] > self.lowfreq_threshold:
                    if n_left_hash_funcs > 0:
                        self.hash_relookups += 1
                        continue # try other hash positions
                    self.hash_collisions += 1
                else: # set new ngram or overwrite low-freq ngram
                    if self.ht[i][0] > 0:
                        self.hash_overwrites += 1
                        self.hash_counter_lost += self.ht[i][0]
                    else:
                        self.hash_added_keys += 1
                    self.ht[i] = (1, ngram)
            break
