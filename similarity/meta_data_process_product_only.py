import gzip
import json
from tqdm import tqdm
import os
import gensim
from gensim import corpora
import re
import pdb

META_ROOT = '/mnt/disk00/asachan/RecFormer/meta_root' # Set your meta data path


'''
pretrain_categories = ['All_Beauty',
'Industrial_and_Scientific',
'AMAZON_FASHION',
'Kindle_Store',
'Appliances',
'Luxury_Beauty',
'Arts_Crafts_and_Sewing',
'Magazine_Subscriptions',
'Automotive',
'Movies_and_TV',
'Books',
'Musical_Instruments',
'CDs_and_Vinyl',
'Office_Products',
'Cell_Phones_and_Accessories',
'Patio_Lawn_and_Garden',
'Clothing_Shoes_and_Jewelry',
'Pet_Supplies',
'Digital_Music',
'Prime_Pantry',
'Gift_Cards',
'Software',
'Grocery_and_Gourmet_Food',
'Sports_and_Outdoors',
'Home_and_Kitchen']
'''

pretrain_categories = ['AMAZON_FASHION']
pretrain_meta_pathes = [f'{META_ROOT}/meta_{cate}.json.gz' for cate in pretrain_categories]

for path in pretrain_meta_pathes:
    if not os.path.exists(path):
        print(path)
        exit(0)
        
def extract_meta_data(path, meta_data):
    title_length = 0
    total_num = 0
    all_product_title = ''
    with gzip.open(path) as f:
        for line in tqdm(f, ncols=100):
            if total_num > 10000:
                return title_length, total_num, all_product_title
            
            line = json.loads(line)
            if 'brand' not in line.keys():
                continue
            if 'title' not in line.keys():
                continue

            brand = line['brand'].lower()
            title = remove_spec_char(line['title'].lower())

            title_length += len(title.split())
            total_num += 1
            
            attr_dict = dict()
            attr_dict['title'] = title
            
            attr_dict['brand'] = brand

            all_product_title = all_product_title + "\n" + title
            
    return title_length, total_num, all_product_title

def remove_spec_char(org_str):
    clean_str = re.sub('\W+',' ', org_str)
    clean_str = re.sub(r'[0-9]+', '', clean_str)
    
    pattern = r"((?<=^)|(?<= )).((?=$)|(?= ))"
    clean_str = re.sub("\s+", " ", re.sub(pattern, '', clean_str).strip())
    return clean_str

def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))

def removeStopwords(wordlist):
    file_path = 'stopwords.txt'
    stopwords = []
    with open(file_path, 'r') as f:
        stopwords.extend(f.read().lower().splitlines())
    return [w for w in wordlist if w not in stopwords]

if __name__ == "__main__":
    meta_data = dict()
    for path in tqdm(pretrain_meta_pathes, ncols=100, desc=path):
        t_l, t_n, t_p = extract_meta_data(path, meta_data)

        fulllist = t_p.split()
        wordfreq = []
        for w in fulllist:
            wordfreq.append(fulllist.count(w))
        
        wordlist = removeStopwords(fulllist)
        freqdict = wordListToFreqDict(wordlist)
        sorteddict = sortFreqDict(freqdict)
                        
        freq_pretrain_meta_pathes = path + "_freq_count.txt"
        
        with open(freq_pretrain_meta_pathes, 'w') as fp:
            for item in sorteddict:
                fp.write("%s " % str(item))
    