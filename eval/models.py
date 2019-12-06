import os
from typing import List

import numpy as np
import torch
from gensim.models import Word2Vec as W2V
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from eval import metric
from mapping import Mapping


def path_dis(w1: str, w2: str) -> float:
    w1_synsets = wn.synsets(w1)
    w2_synsets = wn.synsets(w2)

    res = 0
    for i in w1_synsets:
        for j in w2_synsets:
            res_t = i.path_similarity(j)
            if res_t is None:
                res_t = 0
            res = max(res_t, res)
    return res


def sr_ouput(fn: str, A: List, B: List) -> float:
    print(fn + ':')
    print('\tspearman:{:.3f}'.format(metric.spearman(A, B)))
    print('\tpearson:{:.3f}'.format(metric.pearson(A, B)))


def read_vec(fn: str, csep: str = ';') -> List:
    rst_word, rst_score = list(), list()
    with open(fn, 'r') as fin:
        for line in fin.readlines():
            line = line.strip().split(sep=csep)
            words = list(map(str.lower, line[:-1]))
            rst_word.append(words)
            rst_score.append(float(line[-1]))
    return rst_word, rst_score


class SRModels(object):
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model_name = self.__class__.__name__
        print(f'Loading `{self.model_name}` Model')
        self.model = {}
        self._load_model()
        print(f'Done!')

    def _load_model(self) -> object:
        return None

    def relatedness(self, w1: str, w2: str) -> float:
        return metric.cosine(self.model[w1], self.model[w2])

    def perform(self, base_dir: str = './data/golden/', gamma: float = 1.0):
        for fnt in os.listdir(base_dir):
            if not fnt.endswith('.csv'):
                continue
            fn = os.path.join(base_dir, fnt)
            golden_words, golden_score = read_vec(fn)
            test_score = []
            for word_pair in golden_words:
                w1, w2 = word_pair[0], word_pair[1]
                test_score.append(self.relatedness(w1, w2, gamma))
            sr_ouput(fnt, golden_score, test_score)


class WNGat(SRModels):
    def __init__(self, model_path, combine_model_path: str):
        super().__init__(model_path)
        self.combine_model_path = combine_model_path
        self.model2 = Glove(self.combine_model_path)

    def _load_model(self) -> object:
        emb_data = torch.load(self.model_path).numpy()
        dictionary = Mapping()
        dictionary.load('./data/embedding/wn_gat/dictionary_1w')
        for idx, word in dictionary.id2token.items():
            self.model[word] = emb_data[idx]

    def relatedness(self, w1, w2, gamma: str = 1.0):
        rls1 = super().relatedness(w1, w2)
        rls2 = self.model2.relatedness(w1, w2)
        return gamma * rls1 + (1.0 - gamma) * rls2


class Word2Vec(SRModels):
    def _load_model(self) -> object:
        self.model = W2V.load(self.model_path).wv

    def relatedness(self, w1: str, w2: str, gamma: float = 1.0) -> float:
        rls1 = super().relatedness(w1, w2)
        rls2 = path_dis(w1, w2)
        return gamma * rls1 + (1.0 - gamma) * rls2


class Glove(SRModels):
    def _load_model(self) -> object:
        self.model = {}
        with open(self.model_path, 'r') as fin:
            for line in tqdm(fin.readlines()):
                line = line.split()
                word = line[0]
                embedding = np.array([float(val) for val in line[1:]])
                self.model[word] = embedding

    def relatedness(self, w1: str, w2: str, gamma: float = 1.0) -> float:
        rls1 = super().relatedness(w1, w2)
        if 1.0 - gamma < 1e-6:
            return rls1
        rls2 = path_dis(w1, w2)
        return gamma * rls1 + (1.0 - gamma) * rls2


if __name__ == '__main__':
    from eval.metric import cosine
    from config import word2vec_model_path, glove_model_path, wn_gat_model_path, glove_model_300d_path

    # model = Word2Vec(word2vec_model_path)
    # print(model.relatedness('automobile', 'car'))

    # model = Glove(glove_model_path)
    # print(model.relatedness('automobile', 'car'))

    model = WNGat(wn_gat_model_path, glove_model_300d_path)
    model.perform(gamma=0.8)
