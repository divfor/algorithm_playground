#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import binascii as ba
import pandas as pd
import sys
import os
import shutil
import time
import numpy as np 
from tqdm import tqdm
import re
import glob
from bloomfilter import *
from tensorboardX import SummaryWriter


writer = SummaryWriter('./logs')
cnt,si = 0, 0
add_ok, add_fail, add_skipped, ngrams = 0, 0, 0, 0

def recreate_folders(*args):
    for path in args:
        os.system('sudo rm -rf %s' % path)
        os.system('sudo mkdir -p %s' % path)

    

add_tmp_ok, add_tmp_fail, cnt_tmp, pre_add_ok, cnt_bf =  0, 0, 0, 0, 1
tmp_bf_list = []
def train_filter_reduce_ngrams(bf, bfname, payloads, skip_filters, n=5, stopping_threshold=100,max_wins = 100):                        

    # global cnt,si
    global add_ok, add_fail, add_skipped, ngrams
    global add_tmp_ok, add_tmp_fail, cnt_tmp, pre_add_ok
    bf_tmp = tmp_bf_list[-1]
    max_unique_ngrams = 256 ** n

    for payload in tqdm(payloads):  
        payload = payload.strip()
        try:
            bts = ba.unhexlify(payload) # type(bts) == bytes
        except:
            pass
        if len(bts) < n:
            continue
        num_wins = len(bts) - n + 1 # packet may be very large
        if num_wins > max_wins:
            num_wins = max_wins
        for i in range(num_wins):
            exists = 0
            ngrams += 1
            for skip_bf in skip_filters:
                if bts[i:i+n] in skip_bf:
                    exists += 1
            if exists == 0:         
                exists_in_tmp = 0  
                for tmp in tmp_bf_list:
                    if bts[i:i+n] in tmp:
                        exists_in_tmp += 1 
                        break
                if exists_in_tmp == 0:
                    if add_tmp_ok == bf_tmp.capacity*(cnt_tmp+1) : 
                        cnt_tmp += 1
                        bf_tmp = open_bloom_filter('tmp_%d.bloom'%cnt_tmp,200000000)
                        tmp_bf_list.append(bf_tmp)
                    if bf_tmp.add(bts[i:i+n]) == False:
                        add_tmp_ok += 1
                    else:
                        add_tmp_fail += 1   #tmp_bf冲突个数，一般为0
                else:
                    if bf.add(bts[i:i+n]) == False:
                        add_ok += 1
                    else:
                        add_fail += 1   
            else:
                add_skipped += 1

            if ngrams % 10000 == 0:
                ptbt = (max_unique_ngrams - add_ok - add_tmp_ok)/max_unique_ngrams*100
                # writer.add_scalars('tmp_%d.bloom'%cnt_tmp,{'add_skipped':add_skipped,'add_ok':add_ok,'add_fail':add_fail,
                #         'add_tmp_ok':add_tmp_ok,'add_tmp_fail':add_tmp_fail,'capacity':bf_tmp.capacity*(cnt_tmp+1)},ngrams)
                writer.add_scalars('%s_n%d'%(bfname[:-6],n),{'add_ok':add_ok,'add_tmp_ok':add_tmp_ok},ngrams)
                writer.add_scalars('%s_n%d_unseen_ngrams_rate'%(bfname[:-6],n),{'ptbt':ptbt},ngrams)
            
                
bf_list = []      
def train_normal_filter_reduce_ngrams(bfname, payloads, skip_filters=[], bad_filters=[],n=5,capacity=100000000,max_wins = 100):
    
    global add_ok, add_fail, add_skipped, ngrams
    global add_tmp_ok, add_tmp_fail, cnt_tmp, cnt_bf, pre_add_ok
    bf_tmp = tmp_bf_list[-1]
    bf = bf_list[-1]
    max_unique_ngrams = 256 ** n

    for payload in tqdm(payloads):
        payload = payload.strip()
        try:
            bts = ba.unhexlify(payload) # type(bts) == bytes
        except:
            continue
        if len(bts) < n:
            continue
        num_wins = len(bts) - n + 1 # packet may be very large
       
        p5threshold = int(0.05 * num_wins)    #0.05
        in_bad_filters = 0
        for i in range(num_wins):
            for bad_bf in bad_filters:
                if bts[i:i+n] in bad_bf:
                    in_bad_filters += 1
                    break
            if in_bad_filters > p5threshold:
                break
        if in_bad_filters > p5threshold:
                continue
        
        if num_wins > max_wins:
            num_wins = max_wins

        for i in range(num_wins):
            exists = 0
            ngrams += 1
            for skip_bf in skip_filters:
                if bts[i:i+n] in skip_bf: 
                    exists += 1         
            if exists == 0:         
                exists_in_tmp = 0  
                for tmp in tmp_bf_list:
                    if bts[i:i+n] in tmp:
                        exists_in_tmp += 1 
                        break
                if exists_in_tmp == 0:
                    if add_tmp_ok == bf_tmp.capacity*(cnt_tmp+1) : 
                        cnt_tmp += 1
                        bf_tmp = open_bloom_filter('tmp_%d.bloom'%cnt_tmp,200000000)
                        tmp_bf_list.append(bf_tmp)
                    if bf_tmp.add(bts[i:i+n]) == False:
                        add_tmp_ok += 1
                    else:
                        add_tmp_fail += 1   #tmp_bf冲突个数，一般为0
                else:
                    
                    if add_ok == capacity*cnt_bf : 
                        cnt_bf += 1
                        bf = open_bloom_filter('normal%d_n%d.bloom'%(cnt_bf,n),capacity)
                        bf_list.append(bf)
                    if bf.add(bts[i:i+n]) == False:
                        add_ok += 1
                    else:
                        add_fail += 1        
            else:
                add_skipped += 1
            if ngrams % 10000 == 0: 
                writer.add_scalars('tmp_%d.bloom'%cnt_tmp,{'add_skipped':add_skipped,'add_ok':add_ok,'add_fail':add_fail,
                        'add_tmp_ok':add_tmp_ok,'add_tmp_fail':add_tmp_fail},ngrams)
                # ptbt = (max_unique_ngrams - add_ok - add_tmp_ok)/max_unique_ngrams*100
                # writer.add_scalars('%s'%(bfname[:-6]),{'add_ok':add_ok,'add_tmp_ok':add_tmp_ok},ngrams)
                # writer.add_scalars('%s_unseen_ngrams_rate'%(bfname[:-6]),{'ptbt':ptbt},ngrams)



def train(dataname,bfname,train_txt_num=123,capacity=100000000,n=5):
    print('Start to train %s, save in ./logs/%s_n%d' % (bfname,bfname[:-6],n))

    datapath = '/data/%s/payloads' % dataname
    if 'gold' in bfname: datapath += '/gold'
    elif 'normal' in bfname: datapath += '/normal'
    usedpath = datapath + '/' + 'used'
    recreate_folders('./logs','/data/bloomfilters/tmp')
    os.system('sudo mkdir -p %s' % usedpath)

    fname_list = [fname for fname in os.listdir(datapath) if 'txt' in fname]
    bf = open_bloom_filter(bfname,capacity)
    bf_list.append(bf)
    
    global add_ok, add_fail, add_skipped, ngrams 
    global add_tmp_ok, add_tmp_fail, cnt_tmp 
    bf_tmp = open_bloom_filter('tmp_%d.bloom'%cnt_tmp,200000000)    
    tmp_bf_list.append(bf_tmp)
    
    
    for i, fname in enumerate(fname_list[:train_txt_num]):  
        payloads = open(datapath+'/'+fname).readlines()
        print(i+1, fname) 
        if 'gold' in bfname:
            train_filter_reduce_ngrams(bf,bfname,payloads,[],n) 
        elif 'normal' in bfname: 
            train_normal_filter_reduce_ngrams(bfname, payloads,get_bloom_filters(['bad','normal'],bfname),get_bloom_filters(['bad'],bfname),n)
        elif 'bad' in bfname:
            train_filter_reduce_ngrams(bf,bfname,payloads,get_bloom_filters(['gold'],bfname),n) 
        os.system('sudo mv -f %s %s/' % (datapath+'/'+fname,usedpath))
    close_bloom_filter(bf,add_skipped=add_skipped,add_ok=add_ok,add_fail=add_fail,add_tmp_ok=add_tmp_ok,
                        add_tmp_fail = add_tmp_fail, ngrams=ngrams) 

    print('End training %s' % bfname)


if __name__ == "__main__":
    # train('data1','gold.bloom',n=5)  # 1230w
    
    # train('bad1_80pct.bloom',0.9)

    train('data1','normal1_n5.bloom',n=5,train_txt_num=228)  # 2280w
    
    
    

    


            
        

