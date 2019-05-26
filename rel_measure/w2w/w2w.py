from gensim.models import Word2Vec
from config import word2vec_model_path

def w2w_rel(w1: str, w2:str) -> float:
    w2w_model = Word2Vec.load(word2vec_model_path)
    return w2w_model.similarity(w1, w2)

if __name__ == "__main__":
    print(w2w_rel('century', 'time'))
