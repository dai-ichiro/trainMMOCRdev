import glob
import pandas as pd

excel_files = glob.glob('*.xlsx')

drugs = set([])

for excel_file in excel_files:
    df = pd.read_excel(excel_file)
    set1 = set(df['成分名'])
    set2 = set(df['品名'])
    total = set([x.replace(' ', '').replace('　', '') for x in (set1 | set2)])
    drugs = drugs.union(total)

with open('text.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(drugs))

with open('dicts.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(set(''.join(drugs))))
