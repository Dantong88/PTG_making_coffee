import numpy as np
import pandas as pd
import json
import os

TRAIN_PLUS_VAL = True


path = '/shared/niudt/detectron2/datasets/lvis/lvis_v1_train_ori.json'
path_val = '/shared/niudt/detectron2/datasets/lvis/lvis_v1_val.json'
with open(path, 'r', encoding='utf-8') as fw:
    injson = json.load(fw)
info = injson['info']
annotations = injson['annotations']
images = injson['images']
license = injson['licenses']
categories = injson['categories']

with open(path_val, 'r', encoding='utf-8') as fw2:
    injson_val = json.load(fw2)
info_val = injson_val['info']
annotations_val = injson_val['annotations']
images_val = injson_val['images']
license_val = injson_val['licenses']
categories_val = injson_val['categories']

# ann_val = []
# for _ann in annotations_val:
#     if _ann in annotations:
#         ann_val.append(_ann)
#
# annotations_val = ann_val


df = pd.read_excel('./EPIC_VS_LVIS.xlsx').values
epic_categories = []

epic_noun_list = pd.read_csv('/shared/niudt/detectron2/process_dataset/EPIC_100_noun_classes.csv').values

for i in range(300):
    noun = epic_noun_list[i, 1]
    if len(noun.split(':')) > 1:
        noun = noun.split(':')[1] + '_' + noun.split(':')[0]
    epic_categories.append(noun)

new_categories = []
for i in range(300):
    cate = {}
    cate['name'] = epic_categories[i]
    cate['instances_count'] = 0
    cate['def'] = ''
    cate['synonyms'] = []
    cate['image_count'] = []
    cate['id'] = i + 1
    cate['frequency'] = ''
    cate['synset'] = ''
    new_categories.append(cate)


x = []
for i in range(300):
    print(i + 1)
    if df[i, 4] == 'NN':
        continue
    names = df[i, 4].split(';')
    ids = str(df[i, 5]).split(';')
    for j in range(len(ids)):
        if not categories[int(ids[j]) - 1]['name'] in x:
            x.append(categories[int(ids[j]) - 1]['name'])
        else:
            print(categories[int(ids[j]) - 1]['name'] + 'is already in ' + df[i, 1])
        if not categories[int(ids[j]) - 1]['name'] == names[j]:
            print(names[j])
            print('something wring with: ', names)
    if len(names) != len(ids):
        print('something wring with: ', names)

id = 0
new_annotations = []
print('annotation example is: ', annotations[0])
print('\n')
for ann in annotations:
    if id == 100:
        break
    for i in range(300):
        names = df[i, 4].split(';')
        ids = str(df[i, 5]).split(';')
        if str(ann['category_id']) in ids:
            id = id + 1
            ann_temp = ann
            ann_temp['id'] = id
            ann_temp['category_id'] = i + 1
            new_categories[i]['instances_count'] += 1
            if not ann['image_id'] in new_categories[i]['image_count']:
                new_categories[i]['image_count'].append(ann['image_id'])
            new_annotations.append(ann_temp)
            break

if TRAIN_PLUS_VAL == True:
    for ann in annotations_val:
        # if id == 200:
        #     break
        for i in range(300):
            names = df[i, 4].split(';')
            ids = str(df[i, 5]).split(';')
            if str(ann['category_id']) in ids:
                id = id + 1
                ann_temp = ann
                ann_temp['id'] = id
                ann_temp['category_id'] = i + 1
                new_categories[i]['instances_count'] += 1
                if not ann['image_id'] in new_categories[i]['image_count']:
                    new_categories[i]['image_count'].append(ann['image_id'])
                new_annotations.append(ann_temp)
                break

for i in range(len(new_categories)):
    # print(new_categories[i]['image_count'])
    # print('\n')
    new_categories[i]['image_count'] = len(new_categories[i]['image_count'])


new_injson = {}
new_injson['info'] = info
new_injson['annotations'] = new_annotations
if TRAIN_PLUS_VAL == True:
    new_injson['images'] = images + images_val
else:
    new_injson['images'] = images
new_injson['licenses'] = license
new_injson['categories'] = new_categories
# print(new_annotations)
print(new_categories)

save_path = './new_json_file'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if TRAIN_PLUS_VAL == True:
    save_dir = os.path.join(save_path, 'new_train_plus_val.json')
else:
    save_dir = os.path.join(save_path, 'new_train_single.json')
with open(save_dir, 'w') as fp:
    json.dump(new_injson, fp)




