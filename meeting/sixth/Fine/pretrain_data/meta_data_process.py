import gzip
import json
from tqdm import tqdm
import os

META_ROOT = 'C:/Users/PC/Desktop/Dataset/small_subset/metadata' # Set your meta data path
SEQ_ROOT = 'C:/Users/PC/Desktop/Dataset/small_subset_of_8_categories' # Set your seq data path

pretrain_categories = ['Automotive', 'Cell_Phones_and_Accessories', \
              'Clothing_Shoes_and_Jewelry', 'Electronics', 'Grocery_and_Gourmet_Food', 'Home_and_Kitchen', \
              'Movies_and_TV', 'CDs_and_Vinyl']

# pretrain_meta_pathes = [f'{META_ROOT}/meta_{cate}.json.gz' for cate in pretrain_categories]
pretrain_meta_pathes = [f'{META_ROOT}/{cate}.json.gz' for cate in pretrain_categories]
pretrain_seq_pathes = [f'{SEQ_ROOT}/{cate}_5.json.gz' for cate in pretrain_categories]

for path in pretrain_meta_pathes+pretrain_seq_pathes:
    if not os.path.exists(path):
        print(path)
        exit(0)

def extract_meta_data(path, meta_data, selected_asins):
    title_length = 0
    total_num = 0
    with gzip.open(path) as f:
        for line in tqdm(f, ncols=100):
            line = json.loads(line)
            attr_dict = dict()
            asin = line['asin']
            if asin not in selected_asins:
                continue
            
            category = ' '.join(line['category'])
            brand = line['brand']
            title = line['title']

            title_length += len(title.split())
            total_num += 1

            attr_dict['title'] = title
            attr_dict['brand'] = brand
            attr_dict['category'] = category
            meta_data[asin] = attr_dict   
    return title_length, total_num    


meta_asins = set()
seq_asins1 = set()
seq_asins2 = set()

for path in tqdm(pretrain_meta_pathes, ncols=100, desc='Check meta asins'):
    with gzip.open(path) as f:
        for line in f:
            line = json.loads(line)
            print('-------------------------')
            print(line)
            print('-------------------------')
            if line['asin'] is not None and 'title' in line and line['title'] is not None:
                meta_asins.add(line['asin'])

for path in tqdm(pretrain_seq_pathes, ncols=100, desc='Check seq asins'):
    with gzip.open(path) as f:
        for line in f:
            line = json.loads(line)
            if line['asin'] is not None and 'reviewerID' in line  and line['reviewerID'] is not None:
                seq_asins1.add(line['asin'])
                if line['unixReviewTime'] is not None:
                    seq_asins2.add(line['asin'])

selected_asins = meta_asins & seq_asins2
print(f'Meta has {len(meta_asins)} Asins.')
print(f'Seq has {len(seq_asins1)} Asins.')
print(f'Seq has {len(seq_asins2)} Asins.')
print(f'{len(selected_asins)} Asins are selected.')

meta_data = dict()
for path in tqdm(pretrain_meta_pathes, ncols=100, desc=path):
    t_l, t_n = extract_meta_data(path, meta_data, selected_asins)
    print(f'Average title length of {path}', t_l/t_n)

def add_timestamp_attribute(path, meta_data, selected_asins):
    with gzip.open(path) as f:
        for line in tqdm(f, ncols=100):
            line = json.loads(line)
            asin = line['asin']
            if asin not in selected_asins:
                continue
            meta_data[asin].update({'time':line['unixReviewTime']}) 
    return   

print(f'Before adding time series', len(meta_data))

for path in tqdm(pretrain_seq_pathes, ncols=100, desc=path):
    add_timestamp_attribute(path, meta_data, selected_asins)

print(f'After adding time series', len(meta_data))

with open('time_series_meta_data.json', 'w', encoding='utf8') as f:
    json.dump(meta_data, f)