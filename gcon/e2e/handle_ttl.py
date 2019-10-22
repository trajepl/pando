import os
from collections import defaultdict
from multiprocessing import Pool
from typing import List

from tqdm import tqdm

# line_num = 183605697
line_num = 10001

w2e_path_prefix = './gcon/w2e/data/'
e2e_path_prefix = './gcon/e2e/data/'
ent_dict_path = os.path.join(w2e_path_prefix, 'entity_map')
page_links_dict_path = os.path.join(e2e_path_prefix, 'page_links_map')
page_links_path = os.path.join(e2e_path_prefix, '1w_page_links_en')
ent_dict = {}


def load_ent_dict(sep: str = '\t'):
    print('loading entity map...')
    with open(ent_dict_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin.readlines()):
            line = line.strip().split(sep)
            ent_dict[line[0]] = line[1]


def parse_ttl(ttls: List):
    page_links_dict = defaultdict(set)
    for ttl in tqdm(ttls):
        ttl = ttl.strip().split(' ')
        if ttl[0] in ent_dict.keys() and ttl[1] in ent_dict.keys():
            k = ent_dict[ttl[0]]
            v = ent_dict[ttl[1]]
            page_links_dict[k].add(v)
    return page_links_dict


def paraller_parse_ttl():
    load_ent_dict()
    # ent_dict = {}

    ttls = []
    print('reading raw page links en...')
    with open(page_links_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin.readlines()):
            ttls.append(line)

    cpu_cnt = os.cpu_count()
    step = max(1, int(line_num / cpu_cnt))
    print(f'building mapping among entities on {cpu_cnt} cpus')
    pool = Pool(processes=cpu_cnt)
    rls = []
    for i in range(0, line_num, step):
        rls.append(pool.apply_async(
            func=parse_ttl,
            args=([ttls[i: i+step]])
        ))
    pool.close()
    pool.join()
    page_links_dict = {}
    for item in rls:
        for k, v in item.get().items():
            if k in page_links_dict.keys():
                page_links_dict[k].union(v)
            else:
                page_links_dict[k] = v
    print('Final recording...')
    with open(page_links_dict_path, 'w', encoding='utf-8') as fout:
        for k, v in tqdm(page_links_dict.items()):
            fout.write(k + ' ' + ' '.join(list(v)) + '\n')


if __name__ == "__main__":
    paraller_parse_ttl()