import os
import torch

from tqdm import tqdm

from alldataset import RetrieverDataset, collate_retriever,prepare_sample
from retriever import Retriever


@torch.no_grad()
def main(args):
    device = torch.device(f'cuda:1')

    cpt = torch.load(args.path, map_location='cpu')

    infer_set = RetrieverDataset(dataset_name ="cwq",split='test', skip_no_path=False)

    emb_size = infer_set[0]['q_emb'].shape[-1]
    retriever_config = {
        'topic_pe': True,
        'SPE_kwargs': {
            'num_rounds': 2,
            'num_reverse_rounds': 2
        }
    }
    model = Retriever(emb_size,topic_pe=retriever_config['topic_pe'],SPE_kwargs=retriever_config['DDE_kwargs']).to(device)
    model.load_state_dict(cpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    pred_dict = dict()
    for i in tqdm(range(len(infer_set))):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            relation_embs, topic_entity_one_hot, \
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list']
        relation_list = raw_sample['relation_list']
        top_K_triples = []
        target_relevant_triples = []

        if len(h_id_tensor) != 0:
            pred_triple_logits = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                relation_embs, topic_entity_one_hot)
            pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)
            # 生成随机索引代替TopK
            num_triples = len(pred_triple_scores)
            K = min(args.max_K, num_triples)

            # 创建随机排列并取前K个
            random_indices = torch.randperm(num_triples)[:K]

            top_K_scores = pred_triple_scores[random_indices].cpu().tolist()
            top_K_triple_IDs = random_indices.cpu().tolist()

            for j, triple_id in enumerate(top_K_triple_IDs):
                top_K_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                    top_K_scores[j]
                ))
            # top_K_results = torch.topk(pred_triple_scores,
            #                            min(args.max_K, len(pred_triple_scores)))
            # top_K_scores = top_K_results.values.cpu().tolist()
            # top_K_triple_IDs = top_K_results.indices.cpu().tolist()
            #
            # for j, triple_id in enumerate(top_K_triple_IDs):
            #     top_K_triples.append((
            #         entity_list[h_id_tensor[triple_id].item()],
            #         relation_list[r_id_tensor[triple_id].item()],
            #         entity_list[t_id_tensor[triple_id].item()],
            #         top_K_scores[j]
            #     ))
            #
            # target_relevant_triple_ids = raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist()
            # for triple_id in target_relevant_triple_ids:
            #     target_relevant_triples.append((
            #         entity_list[h_id_tensor[triple_id].item()],
            #         relation_list[r_id_tensor[triple_id].item()],
            #         entity_list[t_id_tensor[triple_id].item()],
            #     ))

        sample_dict = {
            'question': raw_sample['question'],
            'scored_triples': top_K_triples,
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_relevant_triples
        }

        pred_dict[raw_sample['id']] = sample_dict
        if i ==1:
            print(sample_dict['question'])
            print(sample_dict['q_entity'])
            print(sample_dict['scored_triples'])
            print(sample_dict['target_relevant_triples'])


    root_path = os.path.dirname(args.path)
    torch.save(pred_dict, os.path.join(root_path, f'retrieval_cwq_random_{args.max_K}.pth'))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default = "data_files/webqsp/cpt_best.pth",
                        help='Path to a saved model checkpoint')
    parser.add_argument('--max_K', type=int, default=5,
                        help='K in top-K triple retrieval')
    args = parser.parse_args()

    main(args)
