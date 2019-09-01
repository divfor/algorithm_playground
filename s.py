#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import numpy as np
from pybloomfilter import BloomFilter
from cityhash import CityHash64WithSeeds as cityhash
import os
import re
import tqdm

class BloomFilterList(object):
    def __init__(self, capacity, error_rate=1/256, filename_template='bloom_filter_%d.bloom'):
        self.filename_template = filename_template
        self.capacity = capacity
        self.error_rate = error_rate
        self.add_ok = 0
        self.add_fail = 0
        self.bflist_capacity = 0
        self.bflist = []
        self.bflist_file_id = 0 # next file_id to be created
    
    def new_bloom_filter(self):
        filename = self.filename_template % self.bflist_file_id
        bloom = BloomFilter.open(self.capacity, self.error_rate, filename)
        if bloom:
            self.bflist.append(bloom)
            self.bflist_capacity += self.capacity
            self.bflist_file_id += 1
            return bloom
        else:
            print("failed to create new bloom filter")
            return None
    
    def add(self, item):
        if self.add_ok >= self.bflist_capacity:
            self.new_bloom_filter()
        if self.bflist[-1].add(item) == False:
            self.add_ok += 1
        else:
            self.add_fail += 1

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

def cmdget(cmd):
   return [f.strip() for f in os.popen(cmd).readlines()]

def count_in_bloom_filters(item, bf_list):
    return sum([item in bf for bf in bf_list])

def item_in_bloom_filters(item, bf_list):
    return any([item in bf for bf in bf_list])


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
        self.hash_seeds2 = [1768623688, 5213376251, 1536002587, 2694018504, 9277054995,
        4773248795, 4874301328, 2082648990]
 
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
            self.ht = np.load(dumpfile="hash_counters.npy")
        except:
            pass
        if self.ht:
            self.hash_dtype = self.ht.dtype
            self.hash_size = len(self.ht)
        return self.ht

    def save(self, dumpfile="hash_counters.npy"):
        np.save(dumpfile, self.ht)

    def add(self, ngram): # bytes type
        self.hash_add_tries += 1
        n_left_hash_funcs = self.hash_funcs_num
        for seed, seed2 in zip(self.hash_seeds, self.hash_seeds2):
            i = cityhash(ngram.hex(), seed, seed2) % self.hash_size
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



class BytesNgram(object):
    def __init__(self, n = 5, c = 'u2', max_wins = 100, 
    hash_dumpfile = "hash_counters.npy",
    bloom_capacity = 10**8,
    bloom_seen_split_capacity = 2 * 10**8,
    ):
        self.n = n # size for n-gram, default 5 bytes
        self.c = c # size for couter, default 2 bytes
        self.max_wins = max_wins
        self.bloom_capacity = bloom_capacity
        self.bloom_seen_split = bloom_seen_split_capacity
        self.add_ok = 0
        self.add_fail = 0
        self.add_skipped = 0
        self.add_seen_ok = 0
        self.add_seen_fail = 0
        self.add_seen_skipped = 0
        self.seen_full = 0
        self.skip_bflist = []
        self.seen_bflist = []
        self.gold_bloom_files = cmdget("ls gold*.bloom")
        self.bad_bloom_files = cmdget("ls bad*.bloom")
        self.normal_bloom_files = cmdget("ls normal*.bloom")
        self.dt = np.dtype([('counter', self.c), ('n-gram', bytes, self.n)])
        self.h = HashTop(hash_dumpfile, 0, 65535, 7+10**8, dt)
        self.skip = BloomFilterList(self.bloom_capacity, 1/256, 'normal_%d.bloom')
        self.seen = BloomFilterList(self.bloom_capacity, 1/256, 'seen_%d.bloom')


    def tail_seen_bloom(self):
        if self.add_seen_ok >= self.seen_full:
            filename = 'tmp_%d.bloom' % (len(self.seen_bflist) + 1)
            bloom = open_bloom_filter(filename, self.bloom_seen_split)
            self.seen_bflist.addpend(bloom)
            self.seen_full += bloom.capacity
        return self.seen_bflist[-1]

   
    def train_bloom_filter(self, train_bloom, payloads):
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
                ng = bts[i:i+n]
                if self.item_in_bloom_filters(ng, self.skip.bflist):
                    self.h.add(ng)
                    self.add_skipped += 1
                    continue
                if self.item_in_bloom_filters(ng, self.seen.bflist):
                    self.h.add(ng)
                    if train_bloom.add(ng) == False:
                        self.add_ok += 1
                    else:
                        self.add_fail += 1
                    continue
                if tail_seen_bloom().add(ng) == False:
                    self.add_seen_ok += 1
                else:
                    self.add_seen_fail += 1 # should be almost 0

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
    ngp = NgramPacket()
    # train('data1','gold.bloom',n=5)  # 1230w
    # train('bad1_80pct.bloom',0.9)
    ngp.train('data1','normal1_n5.bloom',n=5,train_txt_num=228) # 2280w

   
if __name__ == "__main__":
    main()
