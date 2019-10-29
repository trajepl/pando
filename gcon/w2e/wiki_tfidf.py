import heapq
import pickle
from collections import defaultdict

import gensim
from gensim.corpora import Dictionary, HashDictionary, MmCorpus, WikiCorpus
from gensim.models import TfidfModel
from tqdm import tqdm

wiki_dump_path = '/data/jpli/wikipedia/wiki_tfidf/'

dictionary = Dictionary.load_from_text(wiki_dump_path + '_wordids.txt.bz2')
mm = MmCorpus(wiki_dump_path + '_bow.mm')
tfidf = MmCorpus(wiki_dump_path + '_tfidf.mm')

docno2metadata = None
with open(wiki_dump_path + "_bow.mm.metadata.cpickle", 'rb') as meta_file:
    docno2metadata = pickle.load(meta_file)

word_doc_tfidf = defaultdict(list)
print('word to document tfidf matrix:')
cnt = 0
for i in range(tfidf.num_docs):
    if i % 100000 == 0:
        print(f'complete {i}.')
    tt = sorted(tfidf[i], key=lambda x: x[-1], reverse=True)
    for item in tt[:100]:
        heapq.heappush(word_doc_tfidf[item[0]], (i, item[1]))

# for word, docs in word_doc_tfidf.items():
#     topk = heapq.nlargest(5, docs, key=lambda x: x[1])
#     for i, _ in topk:
#         print(docno2metadata[i], ending='')
#     print()

fout = open(wiki_dump_path + '_r_tfidf.bin', 'wb')
pickle.dump(word_doc_tfidf, fout)
fout.close()
