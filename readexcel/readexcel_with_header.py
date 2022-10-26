from fnmatch import translate
import glob
import pandas as pd
import copy

ZEN = "".join(chr(0xff01 + i) for i in range(94))
HAN = "".join(chr(0x21 + i) for i in range(94))
HAN2ZEN = str.maketrans(HAN, ZEN)

excel_files = glob.glob('*.xlsx')

drugs = set([])

for excel_file in excel_files:
    df = pd.read_excel(excel_file)
    set1 = set(df['成分名'])
    set2 = set(df['品名'])
    total = set([x.replace(' ', '').replace('　', '') for x in (set1 | set2)])
    drugs = drugs.union(total)

all_names = copy.deepcopy(drugs)

for i in range(1, 13):
    with_header = set([f"［{str(i).translate(HAN2ZEN)}］{x}" for x in drugs])
    all_names = all_names.union(with_header)

with open('text.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_names))

with open('dicts.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(set(''.join(all_names))))
