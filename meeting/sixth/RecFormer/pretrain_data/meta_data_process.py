import gzip
import json
from tqdm import tqdm
import os

META_ROOT = '' # Set your meta data path
SEQ_ROOT = '' # Set your seq data path

pretrain_categories = ['Automotive', 'Cell_Phones_and_Accessories', \
              'Clothing_Shoes_and_Jewelry', 'Electronics', 'Grocery_and_Gourmet_Food', 'Home_and_Kitchen', \
              'Movies_and_TV', 'CDs_and_Vinyl']

pretrain_meta_pathes = [f'{META_ROOT}/meta_{cate}.json.gz' for cate in pretrain_categories]
pretrain_seq_pathes = [f'{SEQ_ROOT}/{cate}_5.json.gz' for cate in pretrain_categories]

for path in pretrain_meta_pathes+pretrain_seq_pathes:
    if not os.path.exists(path):
        print(path)
        exit(0)

def extract_meta_data(path, meta_data, selected_asins): # 从给定路径的文件中提取元数据,并将其存储到meta_data字典中
    title_length = 0
    total_num = 0
    with gzip.open(path) as f:                  # 使用gzip.open()函数打开给定路径的文件,并使用with语句进行上下文管理,确保文件在使用后正确关闭
        for line in tqdm(f, ncols=100):         # 使用tqdm库中的进度条迭代文件的每一行
            line = json.loads(line)             # 对于每一行,先将其解析为JSON格式,通过json.loads(line)实现
            attr_dict = dict()                  # 创建一个空字典attr_dict,用于存储当前行的属性
            asin = line['asin']                 # Amazon 产品 ASIN 码
            if asin not in selected_asins:      # (跳过当前行的处理,继续下一行的迭代) 对于每一条产品评价数据,首先检查这条评价对应的产品asin是否在selected_asins列表中。如果不在列表中,说明这条评价信息不属于我们感兴趣和提取的产品,那么就会使用continue关键字跳过当前循环,直接进入下一个评价数据的循环
                continue
            
            # 只处理符合要求的数据
            category = ' '.join(line['category'])
            brand = line['brand']
            title = line['title']

            # 将产品完整标题字符串拆分开,方便后续进行词频统计,关键词提取等文本分析需要
            title_length += len(title.split())  # 将标题（title）按空格分割 (按空格拆分开,形成一个词列表),并将其长度加到title_length上 
            total_num += 1                      # 增加total_num的值,表示处理了一个标题

            # 将title、brand和category存储到attr_dict字典中
            attr_dict['title'] = title
            attr_dict['brand'] = brand
            attr_dict['category'] = category
            meta_data[asin] = attr_dict   
    return title_length, total_num    

meta_asins = set()
seq_asins = set()

for path in tqdm(pretrain_meta_pathes, ncols=100, desc='Check meta asins'): # 从多个文件中提取产品asin码,存放到一个set集合中,目的是去重和去除None值
    with gzip.open(path) as f:
        for line in f:
            line = json.loads(line)
            if line['asin'] is not None and line['title'] is not None:
                meta_asins.add(line['asin'])

for path in tqdm(pretrain_seq_pathes, ncols=100, desc='Check seq asins'):
    with gzip.open(path) as f:
        for line in f:
            line = json.loads(line)
            if line['asin'] is not None and line['reviewerID'] is not None:
                seq_asins.add(line['asin'])

selected_asins = meta_asins & seq_asins                         # 取两个集合的交集
print(f'Meta has {len(meta_asins)} Asins.')
print(f'Seq has {len(seq_asins)} Asins.')
print(f'{len(selected_asins)} Asins are selected.')             # 核实和对比不同来源Asin集合,通过交集数量把握后续工作范围,给后续工作提供参考依据

meta_data = dict()
for path in tqdm(pretrain_meta_pathes, ncols=100, desc=path):   # ncols=100: 指定 tqdm 条状进度条的宽度,设置为100表示进度条占满整行
    t_l, t_n = extract_meta_data(path, meta_data, selected_asins)
    print(f'Average title length of {path}', t_l/t_n)           # 统计和计算产品标题的平均长度：t_l表示总的标题长度和（每个标题长度的求和）,t_n表示总的标题数量
    # 平均长度 = 总长度和 / 总数量

with open('meta_data.json', 'w', encoding='utf8') as f:         # 将meta_data字典写入到文件meta_data.json中
    json.dump(meta_data, f)

"""
1. 挑选出 metadata (product info) 和 samll subset (review info) 中共同的 asin, 放入集合中去重
2. 利用第一步缩小的 asin 范围, 从 metadata 文件中提取每个 asin 产品的相关属性字段 (title, brand, category), 收集到字典 meta_data {asin: attribute dict} 中
3. 将提取出来的 asin meta data 持久化保存到JSON文件, 为后续分析做准备
"""