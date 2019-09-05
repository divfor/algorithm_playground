#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os
import re
import tqdm
import numpy as np
# pip3 install cython; 
# pip3 install https://github.com/divfor/pybloomfiltermmap3/archive/master.zip
from pybloomfilter import BloomFilter
# pip3 install cityhash, see https://github.com/escherba/python-cityhash
# https://github.com/google/cityhash
from cityhash import CityHash64WithSeed as cityhash
# pip3 install datasketch, see https://github.com/ekzhu/datasketch
from datasketch import HyperLogLogPlusPlus
# pip3 install bounter, see https://github.com/RaRe-Technologies/bounter
from bounter import bounter


class BloomFilterDB(BloomFilter):
    def __init__(self, dbfile=None):
        self.dbfile = dbfile

    def load(self):
	try:
	    bf = BloomFilter.open(self.dbfile)
	    meta = json.loads(self.dbfile.replace('.bloom','.meta',1))
	    
    def save():
	pass

class BloomFilterList(object):
    def __init__(self, capacity, error_rate=1/256, filename_template='bloom_filter_%d.bloom'):
        self.filename_template = filename_template
        self.capacity = capacity
        self.error_rate = error_rate
        self.bflist_add_ok = 0
        self.bflist_add_in = 0
        self.bflist_capacity = 0
        self.bflist = []
        self.bflist_next_file_id = 0
    
    def new_bloom_filter(self):
        filename = self.filename_template % self.bflist_next_file_id
        bloom = BloomFilter.open(self.capacity, self.error_rate, filename)
        if not bloom:
            print("failed to create new bloom filter %s" % filename)
            return None
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
	bflist_dbfile_search = self.filename_template.replace('%d', '*', 1)
	bflist_meta = self.filename_template.replace('_%d.bloom', '.meta', 1)
	dbfiles_ids = cmdget("ls %s |awk -F_ '{print $NF}' |awk -F. '{print $1}'" % bflist_dbfile_search)
	self.bflist_next_file_id = max([int(i) for i in dbfiles_ids]) + 1
        dbfiles = cmdget("ls %s | sort" % bflist_dbfile_search)
	for f in dbfiles:
	    meta = f.replace('.bloom','.meta',1)
	    try:
                bf = BloomFilter.open(f)
		bflist_len = cmdget("cat %s" % self.dbfile.replace('.bloom','.meta',1))[0]
	    except:
		print("Failed to load bloom filter %s" % f)
            self.bflist.append(bf)
            self.bflist_capacity += bf.capacity
	    self.bflist_add_ok += int(bflist_len)
	    

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
    bf.sync()        Ω    
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
        self.gold = BloomFilterList(self.gold_capacity, 1/256, 'gold_%d.bloom')
        self.normal = BloomFilterList(self.normal_capacity, 1/256, 'normal_%d.bloom')
        self.bad = BloomFilterList(self.bad_capacity, 1/256, 'bad_%d.bloom')
        self.seen = BloomFilterList(self.seen_capacity, 1/256, 'seen_%d.bloom')
	self.bnt = bounter(need_counts=False) # use HLL algorithm only
	self.bnt_estimated__total_ngrams = 0
	self.hll = HyperLogLogPlusPlus(p=14)
	self.hll_estimated__total_ngrams = 0
   
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
