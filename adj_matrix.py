# import torch
from trankit import Pipeline
from dataset import load_sentence_data
from tqdm import tqdm
import pickle
import numpy as np

p = Pipeline(lang='english', gpu=True, cache_dir='./cache')

def get_adj_matrix(sent):
    tagged_sent = p.posdep(sent, is_sent=True)
    tokens = tagged_sent['tokens']
    sub_adj_matrix = np.eye(len(sent), dtype=np.float32)
    for token in tokens:
        u,v = token['id'] - 1 , token['head'] - 1
        sub_adj_matrix[u][v] = 1.0
    return sub_adj_matrix

if __name__ == '__main__':
    names = ['train', 'dev', 'test']
    for name in names:
        path = f'data/{name}.jsonlines'
        sentence_data = load_sentence_data(path)
        adj_matrix = []

        for i, item in enumerate(tqdm(sentence_data, desc=f'Converting {name} to adj matrix')):
            sent, labels = item
            sub_adj_matrix = get_adj_matrix(sent)
            adj_matrix.append(sub_adj_matrix)

        out_path = f'data/adj_matrix/{name}.mat'
        with open(out_path, 'wb') as f:
            pickle.dump(adj_matrix, f)

