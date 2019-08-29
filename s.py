#!/usr/bin/env python3

def count_item_in_bloom_filters(item, bf_list):
    return sum([item in bf for bf in bf_list])

def item_in_bloom_filters(item, bf_list):
    for bf in bf_list:
        if item in bf:
            return True
    return False

def last_bloom_filter(bf_list, total_added):
    #if total_added > sum([bf.capacity for bf in bf_list]):
    if total_added > bf.capacity * len(bf_list):
        bf = open_bloom_filter('tmp_%d.bloom' % (len(bf_list)+1), 200000000)
        bf_list.append(bf)
    return bf_list[-1]

def inc_hash_counter(ht, ngram):
    for f in HASHFUNCS:
        hv = f(ngram)
        i = hv % HASHLEN
        if ht[i][0] == 0:
            ht[i][0] = 2
            ht[i][1] = ngram
            break
        else:
            if ht[i][1] == ngram:
                ht[i][0] += 1
                break
            else:
                hash_collision += 1

def train_filter_reduce_ngrams(bf, bfname, payloads, skip_filters, n=5, stopping_threshold=100,max_wins = 100):                        
    # global cnt,si
    global add_ok, add_fail, add_skipped, ngrams
    global add_tmp_ok, add_tmp_fail, cnt_tmp, pre_add_ok
    bf_tmp = tmp_bf_list[-1]
    max_unique_ngrams = 256 ** n

    ht = np.zeros(10**8, dtype=np.dtype([('count', '<i2'), ('ngram', np.byte, n)]))
    #ht.dump("filename.db")

    for payload in tqdm(payloads):
        try:
            bts = ba.unhexlify(payload.strip()) # type(bts) == bytes
        except:
            pass
        if len(bts) < n:
            continue
        num_wins = len(bts) - n + 1 # packet may be very large
        if num_wins > max_wins:
            num_wins = max_wins
        for i in range(num_wins):
            ngrams += 1
            ng = bts[i:i+n]
            if item_in_bloom_filters(ng, skip_filters):
                inc_hash_counter(ht, ng)
                add_skipped += 1
                contiune

            if item_in_bloom_filters(ng, tmp_bf_list):
                inc_hash_counter(ht, ng)
                if bf.add(ng) == False:
                    add_ok += 1
                else:
                    add_fail += 1
              
            if last_bloom_filter(tmp_bf_list, add_tmp_ok).add(ng) == False:
                add_tmp_ok += 1
            else:
                add_tmp_fail += 1 #tmp_bf冲突个数，一般为0


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
    
