from typing import Dict

import numpy as np
import torch
from gensim.models import Word2Vec
from tqdm import tqdm

from mapping import Mapping


class ModelLoad():
    def __init__(self, model_name: str, model_path: str) -> None:
        self.model = {}
        self.model_name = model_name
        self.model_path = model_path

    def load(self) -> Dict:
        print(f'Loading {self.model_name} Model')
        if self.model_name.startswith('word2vec'):
            self.model = Word2Vec.load(self.model_path).wv
        elif self.model_name == 'wn_gat':
            emb_data = torch.load(self.model_path).numpy()
            dictionary = Mapping()
            dictionary.load('./data/embedding/wn_gat/dictionary_1w')
            for idx, word in dictionary.id2token.items():
                self.model[word] = emb_data[idx]
        elif self.model_name.startswith('glove'):
            with open(self.model_path, 'r') as fin:
                for line in tqdm(fin.readlines()):
                    line = line.split()
                    word = line[0]
                    embedding = np.array([float(val) for val in line[1:]])
                    self.model[word] = embedding
            print('Done.', len(self.model), 'words loaded!')


if __name__ == '__main__':
    from eval.metric import cosine
    from config import wn_gat_model_path
    model = ModelLoad('wn_gat', wn_gat_model_path)
    model.load()
    print(cosine(model.model['automobile'], model.model['car']))
