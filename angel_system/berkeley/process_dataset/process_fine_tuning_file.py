import numpy as np
import pandas as pd
import json
import os
import time



path_val = '/shared/niudt/detectron2/datasets/lvis/lvis_v1_val.json'
with open(path_val, 'r', encoding='utf-8') as fw2:
    injson_val = json.load(fw2)
info_val = injson_val['info']
annotations_val = injson_val['annotations']
images_val = injson_val['images']
license_val = injson_val['licenses']
categories_val = injson_val['categories']



path = '/shared/niudt/detectron2/process_dataset/ft_images/making_coffee_1_all/result.json'
with open(path, 'r', encoding='utf-8') as fw:
    injson = json.load(fw)

info = injson['info']
annotations = injson['annotations']
images = injson['images']
license = license_val
categories = injson['categories']

# process
s = 1

new_categories = []
for cate in categories:
    cate_cur = {}
    cate_cur['name'] = cate['name']
    cate['instances_count'] = 0
    cate['def'] = ''
    cate['synonyms'] = []
    cate['image_count'] = []
    cate['id'] = cate['id'] + 1
    cate['frequency'] = ''
    cate['synset'] = ''
    new_categories.append(cate)

new_images = []
for image in images:
    image_cur = {}
    image_cur['width'] = image['width']
    image_cur['height'] = image['height']
    image_cur['id'] = image['id'] + 1
    image_cur['file_name'] = image['file_name'].split('/')[-1]
    new_images.append(image_cur)

new_annotations = []
for ann in annotations:
    ann_cur = {}
    ann_cur['area'] = ann['area']
    ann_cur['id'] = ann['id'] + 1
    ann_cur['image_id'] = ann['image_id'] + 1
    if ann['image_id'] + 1 not in new_categories[ann['category_id']]['image_count']:
        new_categories[ann['category_id']]['image_count'].append(ann_cur['image_id'])
    ann_cur['category_id'] = ann['category_id'] + 1
    new_categories[ann['category_id']]['instances_count'] += 1
    ann_cur['segmentation'] = []
    ann_cur['bbox'] = ann['bbox']
    new_annotations.append(ann_cur)

for i in range(len(new_categories)):
    # print(new_categories[i]['image_count'])
    # print('\n')
    new_categories[i]['image_count'] = len(new_categories[i]['image_count'])



new_injson = {}
new_injson['info'] = info
new_injson['annotations'] = new_annotations
new_injson['images'] = new_images
new_injson['licenses'] = license
new_injson['categories'] = new_categories
# print(new_annotations)
print(new_categories)

save_root = time.strftime("%Y-%m-%d %X").split(' ')[0]
save_path = os.path.join('.', save_root, 'new_json_file')
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_dir = os.path.join(save_path, 'fine-tuning.json')
with open(save_dir, 'w') as fp:
    json.dump(new_injson, fp)







