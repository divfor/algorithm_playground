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

def open_bloom_filter(bfname,capacity):
    savepath = '/data/bloomfilters'
    if 'gold' in bfname: savepath += '/gold'
    elif 'bad' in bfname: savepath += '/bad'
    elif 'normal' in bfname: savepath += '/normal'
    else: savepath += '/tmp'
    if not os.path.exists(savepath): os.system('sudo mkdir -p %s' % savepath)
    if os.path.exists('%s/%s' % (savepath,bfname)):
        t = input("Bloom file %s is exists, whether to clear? Input 'y' or 'n' : " % bfname)
        t = 'n'
        if t == 'y': 
            os.system('sudo rm -rf %s/%s' % (savepath,bfname))
            bf = BloomFilter(capacity, 1/256, '%s/%s' % (savepath,bfname))
        else:
            bf = BloomFilter.open('%s/%s' % (savepath,bfname))
    else:
        bf = BloomFilter(capacity, 1/256, '%s/%s' % (savepath,bfname))
    return bf

def close_bloom_filter(bf,**kwargs):
    print('bf.capacity',bf.capacity)
    print('bf.num_bits',bf.num_bits)
    print('bf.num_hashes',bf.num_hashes)
    print('bf.hash_seeds',bf.hash_seeds)
    print('len(bf)',len(bf))
    if len(kwargs):
        s = ''
        for key, value in kwargs.items():
            s += '%s = %ld,'%(key,value)
        print(s)
    bf.sync()
    bf.close()

def get_bloom_filters(bftype_list,current_bfname=None):
    bfsavepath = '/data/bloomfilters'
    bf_list = []
    if 'gold' in bftype_list:
        for bfname in os.listdir(bfsavepath+'/gold'):
            if bfname == current_bfname or 'bloom' not in bfname: continue
            bf = BloomFilter.open(bfsavepath+'/gold/'+bfname)
            bf_list.append(bf)
    if 'bad' in bftype_list:
        for bfname in os.listdir(bfsavepath+'/bad'):
            if bfname == current_bfname or 'bloom' not in bfname: continue
            bf = BloomFilter.open(bfsavepath+'/bad/'+bfname)
            bf_list.append(bf)
    if 'normal' in bftype_list:
        for bfname in os.listdir(bfsavepath+'/normal'):
            if bfname == current_bfname or 'bloom' not in bfname: continue
            bf = BloomFilter.open(bfsavepath+'/normal/'+bfname)
            bf_list.append(bf)
    if 'tmp' in bftype_list:
         for bfname in os.listdir(bfsavepath+'/tmp'):
            if bfname == current_bfname or 'bloom' not in bfname: continue
            bf = BloomFilter.open(bfsavepath+'/tmp/'+bfname)
            bf_list.append(bf)
    return bf_list 



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
        self.gold = BloomFilterList(self.gold_capacity, 1/256, 'gold/gold_*.bloom')
        self.normal = BloomFilterList(self.normal_capacity, 1/256, 'normal/normal_*.bloom')
        self.bad = BloomFilterList(self.bad_capacity, 1/256, 'bad/bad_*.bloom')
        self.seen = BloomFilterList(self.seen_capacity, 1/256, 'tmp/seen_*.bloom')

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


def train(dataname,bfname,train_txt_num=123,capacity=100000000,n=5):
    global add_ok, add_fail, add_skipped, ngrams 
    global add_tmp_ok, add_tmp_fail
    bf_list, tmp_bf_list = [], []
    print('Start to train %s, save in ./logs/%s_n%d' % (bfname,bfname[:-6],n))
    datapath = '/data/%s/payloads' % dataname
    if 'gold' in bfname:
    	datapath += '/gold'
    if 'normal' in bfname:
	 datapath += '/normal'
    usedpath = datapath + '/' + 'used'
    recreate_folders('./logs','/data/bloomfilters/tmp')
    os.system('sudo mkdir -p %s' % usedpath)

    fnames = [fname for fname in os.listdir(datapath) if 'txt' in fname]
    bf = open_bloom_filter(bfname,capacity)
    bf_list.append(bf)
    for i, fname in enumerate(fnames[:train_txt_num]):  
        payloads = open(datapath+'/'+fname).readlines()
        print(i+1, fname) 
        if 'gold' in bfname:
            train_bloom_filter(bf,bfname,payloads,[],n) 
        elif 'normal' in bfname: 
            train_normal_filter_reduce_ngrams(bfname, payloads,get_bloom_filters(['bad','normal'],bfname),get_bloom_filters(['bad'],bfname),n)
        elif 'bad' in bfname:
            train_bloom_filter(bf,bfname,payloads,get_bloom_filters(['gold'],bfname),n) 
        os.system('sudo mv -f %s %s/' % (datapath+'/'+fname,usedpath))
    close_bloom_filter(bf,add_skipped=add_skipped,add_ok=add_ok,add_fail=add_fail,add_tmp_ok=add_tmp_ok,
                        add_tmp_fail = add_tmp_fail, ngrams=ngrams) 

    print('End training %s' % bfname)


def main():
    bn = BytesNgram()
    # train('data1','gold.bloom',n=5)  # 1230w
    # train('bad1_80pct.bloom',0.9)
    bn.train('data1','normal1_n5.bloom',n=5,train_txt_num=228) # 2280w

   
if __name__ == "__main__":
    main()
