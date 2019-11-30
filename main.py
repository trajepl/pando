import argparse
import os
from pathlib import Path
from typing import List

from config import glove_model_path
from eval import metric
from eval.model_load import ModelLoad
from eval.wordnet import path_dis


def read_vec(fn: str, csep: str = ';') -> List:
    rst_word, rst_score = list(), list()
    with open(fn, 'r') as fin:
        for line in fin.readlines():
            line = line.strip().split(sep=csep)
            words = list(map(str.lower, line[:-1]))
            rst_word.append(words)
            rst_score.append(float(line[-1]))
    return rst_word, rst_score


def sr_ouput(fn: str, A: List, B: List) -> float:
    print(fn + ':')
    print('\tspearman:{:.3f}'.format(metric.spearman(A, B)))
    print('\tpearson:{:.3f}'.format(metric.pearson(A, B)))


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate word semantic relatedness')
    parser.add_argument('--model', '-m', default='glove_wn',
                        help='specific model to compute semantic relatedness')
    parser.add_argument('--model_path', default=glove_model_path,
                        help='path for model loading')
    parser.add_argument('--golden_data', default=str(
        Path('./data/golden/').resolve()), help='path for golden dataset')
    parser.add_argument('--gamma', type=float, default=0.2)

    args = parser.parse_args()
    model = ModelLoad(args.model, args.model_path)
    model.load()

    # measure
    for fnt in os.listdir(args.golden_data):
        if not fnt.endswith('.csv'):
            continue
        fn = os.path.join(args.golden_data, fnt)
        golden_words, golden_score = read_vec(fn)
        test_score = []
        for word_pair in golden_words:
            try:
                rel_val = metric.cosine(
                    model.model[word_pair[0]], model.model[word_pair[1]])
                dis_val = path_dis(word_pair[0], word_pair[1])

                if args.model.endswith('_wn'):
                    rel_val = args.gamma * rel_val + (1.0 - args.gamma) * dis_val

                test_score.append(rel_val)
            except Exception as e:
                print(e)
        sr_ouput(fnt, golden_score, test_score)


if __name__ == "__main__":
    main()
