import os
import torch

from datasets import load_dataset
from tqdm import tqdm
from get_embs import load_sbert,sber_text2embedding


from dataset import EmbInferDataset

input_file = "/home/disk2/csj/reasoning-on-graphs-master/rmanluo/RoG-webqsp/data"
# input_file = "/home/disk2/csj/reasoning-on-graphs-master/rmanluo/RoG-cwq/data"

save_dir = 'data_files/webqsp/processed'
# save_dir = 'data_files/cwq/processed'
os.makedirs(save_dir, exist_ok=True)

train_set = load_dataset(input_file, split='train')
val_set = load_dataset(input_file, split='validation')
test_set = load_dataset(input_file, split='test')

train_set = EmbInferDataset(
    train_set,
    os.path.join(save_dir, 'train.pkl'))

val_set = EmbInferDataset(
    val_set,
    os.path.join(save_dir, 'val.pkl'))

test_set = EmbInferDataset(
    test_set,
    os.path.join(save_dir, 'test.pkl'),
    skip_no_topic=False,
    skip_no_ans=False)

model, tokenizer, device = load_sbert()
print("模型成功加载")
print(model)

def get_emb(subset, save_file):
    emb_dict = dict()
    for i in tqdm(range(len(subset))):
        id, q_text, text_entity_list, relation_list = subset[i]

        q_emb= sber_text2embedding(model, tokenizer, device, q_text,batch_size =256)
        entity_embs = sber_text2embedding(model, tokenizer, device, text_entity_list,batch_size=256)
        relation_embs =sber_text2embedding(model, tokenizer, device, relation_list,batch_size=256)
        print(q_emb.shape)
        print(entity_embs.shape)
        print(relation_embs.shape)
        emb_dict_i = {
            'q_emb': q_emb,
            'entity_embs': entity_embs,
            'relation_embs': relation_embs
        }
        emb_dict[id] = emb_dict_i

    torch.save(emb_dict, save_file)





emb_save_dir = f'data_files/webqsp/emb/sbert'
# emb_save_dir = f'data_files/cwq/emb/sbert'
os.makedirs(emb_save_dir, exist_ok=True)


get_emb(train_set, os.path.join(emb_save_dir, 'train.pth'))
get_emb(val_set,os.path.join(emb_save_dir, 'val.pth'))
get_emb(test_set, os.path.join(emb_save_dir, 'test.pth'))