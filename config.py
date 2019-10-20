from pathlib import Path

wiki_dump_path = str(
    Path('./data/raw_data/wikipedia/enwiki_page_little.xml.bz2').resolve())
wiki_text_path = str(
    Path('./data/raw_data/wikipedia/enwiki_page_little.txt').resolve())

word2vec_model_path = str(
    Path('./data/embedding/w2w/word2vec.model').resolve())
glove_model_50d_path = str(
    Path('./data/embedding/glove.6B/glove.6B.50d.txt').resolve())
glove_model_100d_path = str(
    Path('./data/embedding/glove.6B/glove.6B.100d.txt').resolve())
glove_model_200d_path = str(
    Path('./data/embedding/glove.6B/glove.6B.200d.txt').resolve())
glove_model_300d_path = str(
    Path('./data/embedding/glove.6B/glove.6B.300d.txt').resolve())
glove_model_path = glove_model_300d_path
