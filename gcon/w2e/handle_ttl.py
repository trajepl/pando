import os
from collections import defaultdict
from multiprocessing import Pool

from tqdm import tqdm

path_prefix = './gcon/w2e/data'
anchor_text_en_ttl = os.path.join(path_prefix, '1w_anchor_text_en.ttl')
ent_dict_path = os.path.join('./gcon/e2e/data', 'ids_page')


# line_num = 163264419

ent_dict = {}
ent_idx = 0
anc_dict = {}
anc_idx = 0
gra_rel = defaultdict(set)


def load_ent_dict(sep: str = ' '):
    print('loading entity map...')
    with open(ent_dict_path, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin.readlines()):
            line = line.strip().split(sep)
            ent_dict[line[1]] = line[0]


def parse_line_ttl(line: str):
    global gra_rel, anc_dict, anc_idx
    line = line.strip().split(' ')
    try:
        line[2] = ' '.join(line[2:-1]).strip().split('"')[1]
        line = [line[0], line[2]]

        if line[0] in ent_dict.keys():
            line[0] = ent_dict[line[0]]

        if not line[1] in anc_dict.keys():
            anc_dict[line[1]] = anc_idx
            anc_idx += 1
        gra_rel[str(anc_dict[line[1]])].add(str(line[0]))
    except Exception as e:
        print(line)
        print(e)


def parse_ttl(fn: str):
    global gra_rel, anc_dict
    load_ent_dict()
    with open(fn, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin.readlines()):
            if line.startswith('#'):
                continue
            parse_line_ttl(line)

    # wirte_graph
    with open(os.path.join(path_prefix, 'entity_anchor_text_map'), 'w', encoding='utf-8') as fout:
        for anc in gra_rel.keys():
            line = anc + ' ' + ' '.join(list(gra_rel[anc]))
            fout.write(line + '\n')
    del gra_rel
    # write entity dict
    with open(os.path.join(path_prefix, 'anchor_text_map'), 'w', encoding='utf-8') as fout:
        for k, v in anc_dict.items():
            line = k + '\t' + str(v)
            fout.write(line + '\n')
    del anc_dict


if __name__ == "__main__":
    parse_ttl(anchor_text_en_ttl)
