import argparse
from pathlib import Path

from config import glove_model_path
from eval import models


parser = argparse.ArgumentParser(
    description='Evaluate word semantic relatedness')
parser.add_argument('--model', '-m', default='glove',
                    help='specific model to compute semantic relatedness')
parser.add_argument('--model_path', default=glove_model_path,
                    help='path for model loading')
parser.add_argument('--golden_data', default=str(
    Path('./data/golden/').resolve()), help='path for golden dataset')
parser.add_argument('--gamma', type=float, default=0.7)

args = parser.parse_args()
if args.model == 'word2vec':
    model = models.Word2Vec(args.model_path)
elif args.model == 'glove':
    model = models.Glove(args.model_path)
elif args.model == 'wngat':
    model = models.WNGat(args.model_path, args.combine_model_path)
else:
    pass

model.perform(gamma=args.gamma)
