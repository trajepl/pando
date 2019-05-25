import base
import multiprocessing 

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from config import wiki_text_path, word2vec_model_path


def train(wiki_text_path, word2vec_model_path):
    model = Word2Vec(
        LineSentence(wiki_text_path),
        size=100,
        window=10,
        min_count=5,
        workers=multiprocessing.cpu_count()
    )
    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)
    model.save(word2vec_model_path)


if __name__ == "__main__":
    train(wiki_text_path, word2vec_model_path)
