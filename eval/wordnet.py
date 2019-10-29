from nltk.corpus import wordnet as wn

def path_dis(w1: str, w2: str) -> float:
    w1_synsets = wn.synsets(w1)
    w2_synsets = wn.synsets(w2)

    res = -100
    for i in w1_synsets:
        for j in w2_synsets:
            res_t = i.path_similarity(j)
            if res_t is None:
                res_t = -100
            res = max(res_t, res)
    return res
