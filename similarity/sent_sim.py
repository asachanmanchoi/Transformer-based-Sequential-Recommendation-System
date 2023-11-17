import torch
import pdb

id =1
torch.cuda.set_device(id)
#torch.device("cpu")
#torch.cuda.set_device()
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/gtr-t5-large')

META_ROOT = '/mnt/disk00/asachan/RecFormer/meta_root' # Set your meta data path

pretrain_categories = ['All_Beauty', 'Industrial_and_Scientific', 'AMAZON_FASHION', 'Kindle_Store',
'Appliances', 'Luxury_Beauty', 'Arts_Crafts_and_Sewing', 'Magazine_Subscriptions', 'Automotive',
'Movies_and_TV', 'Books', 'Musical_Instruments', 'CDs_and_Vinyl', 'Office_Products',
'Cell_Phones_and_Accessories', 'Patio_Lawn_and_Garden', 'Clothing_Shoes_and_Jewelry',
'Pet_Supplies', 'Digital_Music', 'Prime_Pantry', 'Gift_Cards', 'Software',
'Grocery_and_Gourmet_Food', 'Sports_and_Outdoors', 'Home_and_Kitchen']

pretrain_meta_pathes = [f'{META_ROOT}/meta_{cate}.json.gz' for cate in pretrain_categories]
compared_meta_pathes = [f'{META_ROOT}/meta_{cate}.json.gz' for cate in pretrain_categories]

for path in pretrain_meta_pathes:
    compared_meta_pathes = [f'{META_ROOT}/meta_{cate}.json.gz' for cate in pretrain_categories]
    path = path + "_freq.txt"
    while len(compared_meta_pathes) > 0:
        #pdb.set_trace()
        comparing_file_path = compared_meta_pathes.pop(0)
        comparing_file_path += "_freq.txt"

        embedding_1 = None
        embedding_2 = None
        
        with open(path, 'r') as f1:
            embedding_1= model.encode(f1.read(), convert_to_tensor=True)
        with open(comparing_file_path, 'r') as f2:
            embedding_2 = model.encode(f2.read(), convert_to_tensor=True)
        tokenizer = model.tokenizer
        #pdb.set_trace()
        #print (util.pytorch_cos_sim(embedding_1, embedding_2))
        print ("%.5f\t" % round(float(util.pytorch_cos_sim(embedding_1, embedding_2)),5), end="")
        
    print()

    
'''
sentences = ["", ""]



#Compute embedding for both lists
embedding_1= model.encode(sentences[0], convert_to_tensor=True)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)
tokenizer = model.tokenizer


print (util.pytorch_cos_sim(embedding_1, embedding_2))
'''