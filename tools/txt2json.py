import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='text file')
parser.add_argument('--output', type = str, help='json file')
args = parser.parse_args()

input_fname = args.input
output_fname = args.output

with open(input_fname, 'r', encoding='utf-8') as f:
    lines = f.readlines()

lines = [x.strip() for x in lines]

data_list = []
for line in lines:
    img_path, text = line.split(' ')
    data = {
        'img_path': img_path,
        'instances':[{'text':text}]
        }
    data_list.append(data)

result = {
    'metainfo':{
        'dataset_type':'TextRecogDataset',
        'task_name':'textrecog'
    },
    'data_list':data_list
}

with open(output_fname, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)