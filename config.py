from pathlib import Path

wiki_dump_path = str(
    Path('./data/raw_data/wikipedia/enwiki_page_little.xml.bz2').resolve())
wiki_text_path = str(
    Path('./data/raw_data/wikipedia/enwiki_page_little.txt').resolve())

word2vec_model_100d_path = str(
    Path('./data/embedding/word2vec/word2vec_100d.model').resolve())
word2vec_model_path = word2vec_model_100d_path

glove_model_50d_path = str(
    Path('./data/embedding/glove/glove.6B.50d.txt').resolve())
glove_model_100d_path = str(
    Path('./data/embedding/glove/glove.6B.100d.txt').resolve())
glove_model_200d_path = str(
    Path('./data/embedding/glove/glove.6B.200d.txt').resolve())
glove_model_300d_path = str(
    Path('./data/embedding/glove/glove.6B.300d.txt').resolve())
glove_model_42b_300d_path = str(Path('./data/embedding/glove/glove.42B.300d.txt').resolve())
glove_model_path = glove_model_100d_path


wn_gat_450_model_path = str(
    Path('./data/embedding/wn_gat/450_wn_gat.emb').resolve())
wn_gat_model_path = wn_gat_450_model_path

dictionary_path = str(Path('./data/golden/dictionary.txt').resolve())
dictionary_short_path = str(Path('./data/golden/dictionary_short.txt').resolve())
