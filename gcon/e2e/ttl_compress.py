'''
    del relationship(2rd column) in triples
'''
from typing import List

from tqdm import tqdm

from gcon.e2e.categories import Categories
from gcon.e2e.entities import Entities
# from mapping import Mapping
from paraller import ParallerParser

ENTITIES = Entities()
CATEGORIES = Categories()
E2E_PREFIX = './gcon/e2e/data/'

TTLS = []

with open(E2E_PREFIX + 'categories/1w_article_categories_en.ttl', 'r', encoding='utf-8') as fin:
    for line in fin.readlines():
        TTLS.append(line)


class TTLCompress(ParallerParser):
    def parser(self, l, r, **kwargs) -> List[str]:
        rls = []

        for line in tqdm(TTLS[l:r]):
            line = line.strip().split(' ')
            if line[-1] != '.':
                break

            try:
                _ent_idx = ENTITIES.token2id[line[0]]
            except Exception as e:
                _ent_idx = line[0]

            try:
                _cat_idx = CATEGORIES.token2id[line[-2]]
            except Exception as e:
                _cat_idx = line[-2]

            rls.append((_ent_idx, _cat_idx))
        return rls


if __name__ == "__main__":
    parser = TTLCompress(total=len(TTLS))
    rls = parser.run()
    ent_list = []
    for section in rls:
        ent_list += section.get()
    with open(E2E_PREFIX + '/article_categories_unx', 'w') as fout:
        for pair in ent_list:
            fout.write(f'{pair[0]} {pair[1]}\n')
