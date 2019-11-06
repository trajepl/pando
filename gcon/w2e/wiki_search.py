# proxychains4
from typing import List, Optional

# change the function `search` to return `pageid`
import wikipedia
from tqdm import tqdm
from wikipediaapi import Wikipedia, WikipediaPage

from config import dictionary_short_path

wiki = Wikipedia('en')

if __name__ == "__main__":
    words = []
    with open(dictionary_short_path, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            words.append(line.strip())
    with open('./gcon/w2e/data/w2e_no_weight', 'w', encoding='utf-8') as fout:
        for word in tqdm(words):
            fout.write(f'{word};')
            while True:
                try:
                    pages = wikipedia.search(word, results=10)
                    if pages:
                        break
                except Exception as e:
                    print(e, 'retry...')
            str_t = ';'.join([str(page[1]) for page in pages])
            fout.write(str_t)
            fout.write('\n')
