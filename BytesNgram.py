#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os
import re
import json
import tqdm
import numpy as np
# pip3 install cython; 
# pip3 install https://github.com/divfor/pybloomfiltermmap3/archive/master.zip
from pybloomfilter import BloomFilter
from .HashTop import HashTop
from .BloomFilterList import BloomFilterList

class BytesNgram(object):
    '''
        n : num of bytes of one n-gram (always 1-byte sliding)
        c : default 'u2' for unsigned int of 2-bytes from numpy dtype
        max_wins : at most num of n-grams slided from payload, default 100
        gold_capacity : default 10**8, ~137 MB
        bad_capacity : default 10**8, ~137 MB
        normal_capacity : default 10**8, ~137 MB
        seen_capacity : 2*10**8, ~274 MB
    '''
    def __init__(self, n = 5, c = 'u2', max_wins = 100,
                gold_capacity = 10**8,
                bad_capacity = 10**8,
                normal_capacity = 10**8,
                seen_capacity = 2*10**8):
        self.n = n # size for n-gram, default 5 bytes
        self.c = c # size for couter, default 2 bytes
        self.max_wins = max_wins
        self.gold_capacity = gold_capacity
        self.bad_capacity = bad_capacity
        self.normal_capacity = normal_capacity
        self.seen_capacity = seen_capacity
        self.h = None
        self.gold_skipped = 0
        self.gold = None
        self.bad = None
        self.normal = None
        self.seen = None

    def train_gold_bloom_filters(self, payloads):
        gold_hashdb = '.'.join(self.gold.dbfile_base, 'npy')
        dt = np.dtype([('counter', self.c), ('n-gram', bytes, self.n)])
        self.h = HashTop(dumpfile=gold_hashdb, lowfreq_threshold=1, highfreq_threshold=65535,
                        hash_size=self.gold_capacity, hash_dtype=dt)
        self.gold = BloomFilterList(self.gold_capacity, 1/256, 'gold/gold_*.bloom')
        self.seen = BloomFilterList(self.seen_capacity, 1/256, 'tmp/seen_*.bloom')
        self.gold.load()
        self.seen.load()
        for payload in tqdm(payloads):
            try:
                bts = bytes.fromhex(payload.strip()) # ba.unhexlify(hex_string)
            except:
                continue
            if len(bts) < self.n:
                continue
            num_wins = len(bts) - self.n + 1 # packet may be very large
            for i in range(min(num_wins, self.max_wins)):
                self.ngrams += 1
                ng = bts[i:i+n] # bytes type
                self.h.add(ng) # hash counters
                self.gold.add(ng) # bloom filters
                if not any([ng in b for b in self.seen.bflist]):
                    self.seen.add(ng) # bloom filters for one-seen recording
        self.seen.close()
        self.gold.close()
        self.h.close()
   
    def train_bad_bloom_filters(self, payloads):
        bad_hashdb = '.'.join(self.bad.dbfile_base, 'npy')
        dt = np.dtype([('counter', self.c), ('n-gram', bytes, self.n)])
        self.h = HashTop(dumpfile=bad_hashdb, lowfreq_threshold=1, highfreq_threshold=65535,
                        hash_size=self.bad_capacity, hash_dtype=dt)
        self.bad = BloomFilterList(self.bad_capacity, 1/256, 'bad/bad_*.bloom')
        for payload in tqdm(payloads):
            try:
                bts = bytes.fromhex(payload.strip()) # ba.unhexlify(hex_string)
            except:
                continue
            if len(bts) < self.n:
                continue
            num_wins = len(bts) - self.n + 1 # packet may be very large
            for i in range(min(num_wins, self.max_wins)):
                self.ngrams += 1
                ng = bts[i:i+n] # bytes type
                self.h.add(ng) # hash counters
                self.bad.add(ng) # bloom filters
        self.bad.close()
        self.h.close()

    def train_normal_bloom_filters(self, payloads):
        bad_hashdb = '.'.join(self.bad.dbfile_base, 'npy')
        dt = np.dtype([('counter', self.c), ('n-gram', bytes, self.n)])
        self.h = HashTop(dumpfile=bad_hashdb, lowfreq_threshold=2, highfreq_threshold=65535,
                        hash_size=self.normal_capacity, hash_dtype=dt)
        self.normal = BloomFilterList(self.normal_capacity, 1/256, 'normal/normal_*.bloom')
        self.normal.load() # continue work
        self.seen = BloomFilterList(self.seen_capacity, 1/256, 'tmp/seen_*.bloom')
        self.seen.load()
        for payload in tqdm(payloads):
            try:
                bts = bytes.fromhex(payload.strip()) # ba.unhexlify(hex_string)
            except:
                continue
            if len(bts) < self.n:
                continue
            num_wins = len(bts) - self.n + 1 # packet may be very large
            for i in range(min(num_wins, self.max_wins)):
                self.ngrams += 1
                ng = bts[i:i+n] # bytes type
                self.h.add(ng) # hash counters
                self.normal.add(ng) # bloom filters
                if not any([ng in b for b in self.seen.bflist]):
                    self.seen.add(ng) # bloom filters for one-seen recording
        self.normal.close()
        self.seen.close()
        self.h.close()

