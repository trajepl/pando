from tqdm import tqdm
from gensim.corpora import WikiCorpus
from config import wiki_dump_path, wiki_text_path


def make_wiki(wiki_dump_path, wiki_text_path):
    wiki = WikiCorpus(wiki_dump_path)
    with open(wiki_text_path, 'w', encoding='utf-8') as fout:
        for text in tqdm(wiki.get_texts()):
            fout.write(' '.join(text) + '\n')


if __name__ == "__main__":
    make_wiki(wiki_dump_path, wiki_text_path)
