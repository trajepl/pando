from pathlib import Path

wiki_dump_path = str(Path('./data/raw_data/wikipedia/enwiki_page_little.xml.bz2').resolve())
wiki_text_path = str(Path('./data/raw_data/wikipedia1/enwiki_page_little.txt').resolve())

word2vec_model_path = str(Path('./data/embedding/w2w/word2vec.model').resolve())