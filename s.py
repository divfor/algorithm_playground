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
# pip3 install datasketch, see https://github.com/ekzhu/datasketch
from datasketch import HyperLogLogPlusPlus
# pip3 install bounter, see https://github.com/RaRe-Technologies/bounter
from bounter import bounter
from .HashTop import HashTop

def cmdget(cmd):
    return [f.strip() for f in os.popen(cmd).readlines()]

def count_in_bloom_filters(item, bf_list):
    return sum([item in bf for bf in bf_list])

class BloomFilterList(object):
    def __init__(self, capacity, error_rate=1/256, dbfiles='~/gold/gold_*.bloom'):
        self.dbfiles = dbfiles
        self.dbfile_matcher = dbfiles.replace('_*.bloom','_%d.bloom')
        self.metafile_matcher = dbfiles.replace('_*.bloom','_%d.meta')
        self.capacity = capacity
        self.error_rate = error_rate
        self.bflist_add_ok = 0
        self.bflist_add_in = 0
        self.bflist_capacity = 0
        self.bflist_next_file_id = 0
        self.bflist = []

    def save_bloom_filter(self, bloom_filter):
        meta_file = bloom_filter.name.rsplit('.',1)[0] + '.meta'
        cmdget("echo %d > %s" % (len(bloom_filter), meta_file))
        bloom_filter.sync()

    def new_bloom_filter(self):
        filename = self.dbfile_matcher % self.bflist_next_file_id
        bloom = BloomFilter.open(self.capacity, self.error_rate, filename)
        if not bloom:
            print("failed to create new bloom filter %s" % filename)
            return None
        self.save_bloom_filter(bflist[-1])
        self.bflist.append(bloom)
        self.bflist_capacity += self.capacity
        self.bflist_next_file_id += 1
        return bloom

    def add(self, item):
        if self.bflist_add_ok >= self.bflist_capacity:
            self.new_bloom_filter()
        if self.bflist[-1].add(item) == False:
            self.bflist_add_ok += 1
        else:
            self.bflist_add_in += 1

    def load(self): # do not add new bloom filter after load()
        dbfiles_ids = cmdget("ls %s |awk -F_ '{print $NF}' |awk -F. '{print $1}'" % self.dbfiles)
        self.bflist_next_file_id = 1 + max([int(i) for i in dbfiles_ids])
        for f in cmdget("ls %s | sort" % self.dbfiles):
            meta_file = f.rsplit('.',1)[0] + '.meta'
            try:
                b = BloomFilter.open(f)
                n = cmdget("cat %s" % meta_file)[0]
            except:
                print("Failed to load bloom filter %s" % f)
                continue
            self.bflist.append(b)
            self.bflist_capacity += b.capacity
            self.bflist_add_ok += int(n)

    def close(self):
        self.save_bloom_filter(self.bflist[-1])
        for b in self.bflist: 
            summary = str(b).strip('<>') + ', num_bits: %ld' % b.num_bits
            print("\nelements_new_added (%ld): %s\n" % (len(b), b.name))
            print("%s\nHash Seeds: %s\n" % (summary, b.hash_seeds))
            b.close()
        print("\nbflist_capacity: %ld, " % self.bflist_capacity)
        print("bflist_add_repeats: %ld, " % self.bflist_add_in)
        print("bflist_add_ok: %ld\n" % self.bflist_add_ok)


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
    def __init__(self, n = 5, c = 'u2', max_wins = 100, 
    hash_file_template = '%s_counters.npy',
    gold_capacity = 10**8, #137 MB
    bad_capacity = 10**8,
    normal_capacity = 10**8,
    seen_capacity = 2*10**8, #274 MB
    ):
        self.n = n # size for n-gram, default 5 bytes
        self.c = c # size for couter, default 2 bytes
        self.max_wins = max_wins
        self.gold_capacity = gold_capacity
        self.bad_capacity = bad_capacity
        self.normal_capacity = normal_capacity
        self.seen_capacity = seen_capacity
        self.gold_bloom_files = cmdget("ls gold_*.bloom")
        self.bad_bloom_files = cmdget("ls bad_*.bloom")
        self.normal_bloom_files = cmdget("ls normal_*.bloom")
        self.dt = np.dtype([('counter', self.c), ('n-gram', bytes, self.n)])
        self.h = None
        self.hash_file_template = hash_file_template
        self.gold_skipped = 0
        self.gold = BloomFilterList(self.gold_capacity, 1/256, 'gold/gold_*.bloom')
        self.normal = BloomFilterList(self.normal_capacity, 1/256, 'normal/normal_*.bloom')
        self.bad = BloomFilterList(self.bad_capacity, 1/256, 'bad/bad_*.bloom')
        self.seen = BloomFilterList(self.seen_capacity, 1/256, 'tmp/seen_*.bloom')
        self.bnt = bounter(need_counts=False) # use HLL algorithm only
        self.hll = HyperLogLogPlusPlus(p=14)
        self.hll_estimated__total_ngrams = 0
     	self.bnt_estimated__total_ngrams = 0
   
    def train_bad_bloom_filters(self, payloads):
        self.h = HashTop(self.hash_file_template % 'bad', 1, 65535, 7+10**7, self.dt)
        self.gold.load()
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
		utf8ng=ng.hex().encode('utf8')
		self.hll.update(utf8ng)
		self.bnt.update(list(utf8ng))
                if any([ng in b for b in self.gold.bflist]):
                    self.gold_skipped += 1
                elif any([ng in b for b in self.seen.bflist]):
                    self.bad.add(ng)
                else:
                    self.seen.add(ng)
        self.h.save()
	self.hll_estimated__total_ngrams = self.hpp.count()
	self.bnt_estimated__total_ngrams = self.bnt.cardinality()

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
