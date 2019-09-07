#!/usr/bin/env python3
# -*- coding=utf-8 -*-
# pip3 install cython
# pip3 install https://github.com/divfor/pybloomfiltermmap3/archive/master.zip
from pybloomfilter import BloomFilter
import os

class BloomFilterList(object):
    def __init__(self, capacity, error_rate=1/256, dbfiles='~/gold/gold_*.bloom'):
        self.dbfiles = dbfiles
        self.dbfile_base = self.dbfiles.rsplit('_',1)[0] # dbfiles must be like 'xxx_*.bloom'
        self.dbfile_matcher = '_'.join(self.dbfile_base, '%d.bloom')
        self.metafile_matcher = '_'.join(self.dbfile_base, '%d.meta')
        self.capacity = capacity
        self.error_rate = error_rate
        self.bflist_add_ok = 0
        self.bflist_add_in = 0
        self.bflist_capacity = 0
        self.bflist_next_file_id = 0
        self.bflist = []

    def os_shell(cmd):
        return [f.strip() for f in os.popen(cmd).readlines()]
    
    def save_bloom_filter(self, bloom_filter):
        meta_file = bloom_filter.name.rsplit('.',1)[0] + '.meta'
        self.os_shell("echo %d > %s" % (len(bloom_filter), meta_file))
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
        dbfiles_ids = self.os_shell("ls %s |awk -F_ '{print $NF}' |awk -F. '{print $1}'" % self.dbfiles)
        self.bflist_next_file_id = 1 + max([int(i) for i in dbfiles_ids])
        for f in self.os_shell("ls %s | sort" % self.dbfiles):
            meta_file = f.rsplit('.',1)[0] + '.meta'
            try:
                b = BloomFilter.open(f)
                n = self.os_shell("cat %s" % meta_file)[0]
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
        print("bflist_add_ok: %ld\n\n" % self.bflist_add_ok)
