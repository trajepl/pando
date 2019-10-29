import os
from multiprocessing import Pool, Manager
from pathlib import Path

from tqdm import tqdm

page_ids_ttl_fn = Path('./gcon/e2e/data/page_ids_en.ttl').resolve()
ids_page_fn = Path('./gcon/e2e//data/ids_page').resolve()
page_ids_ttl_list = []
res = {}


def parser(ids_page_full_pair, l: int, r: int):
    for line in tqdm(page_ids_ttl_list[l:r]):
        line = line.strip().split(' ')
        idx = line[2].split('^^')[0][1:-1]
        ids_page_full_pair[idx] = line[0]


def paraller_parser():
    line_cnt = 0
    with open(page_ids_ttl_fn, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            if line.startswith('#'):
                continue
            line_cnt += 1
            page_ids_ttl_list.append(line)

    cpu_cnt = os.cpu_count()
    pool = Pool(cpu_cnt)
    step = max(1, line_cnt // cpu_cnt)
    with Manager() as mgr:
        ids_page_full_pair = mgr.dict()
        for i in range(0, line_cnt, step):
            pool.apply_async(func=parser, args=(
                ids_page_full_pair, i, i + step))
        pool.close()
        pool.join()
        res = ids_page_full_pair
        try:
            with open(ids_page_fn, 'w', encoding='utf-8') as fout:
                for k, v in dict(ids_page_full_pair).items(): # cannot iterate the Manager().dict() directly
                    fout.write(f'{k} {v}\n')
        except Exception as e:
            print(e)

if __name__ == "__main__":
    paraller_parser()
