import pandas as pd
import json
import os

TRAIN_PLUS_VAL = True


path = '/shared/niudt/detectron2/datasets/lvis_ori/lvis_v1_train.json'
path_val = '/shared/niudt/detectron2/datasets/lvis_ori/lvis_v1_val.json'
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

ids_list = []
df = pd.read_excel('./cate.xlsx').values


for i in range(1203):
    id = df[i][5]
    name = df[i][2]
    if df[i][1] == 1:
        ids_list.append(id)

mapping_od_id_to_new_id = range(1, len(ids_list) + 1, 1)
new_categories = []
for category in categories:
    if category['id'] in ids_list:
        new_categories.append(category)

# for category in categories_val:
#     if category['id'] in ids_list:
#         for i, _cate in enumerate(new_categories):
#             if category['id'] == _cate['id']:
#                 new_categories[i]['instance_count'] = new_categories[i]['instance_count'] + _cate['instance_count']
#                 break


new_annotations = []
num = 0
for ann in annotations:
    if ann['category_id'] in ids_list:
        num = num + 1
        ann['id'] = num
        new_annotations.append(ann)

num_ann_train = len(new_annotations)

print('number of train ann is : ', num_ann_train)
for ann in annotations_val:
    # if id == 100:
    #     break
    if ann['category_id'] in ids_list:
        num = num + 1
        ann['id'] = num
        new_annotations.append(ann)



for i, cate in enumerate(new_categories):
    old_id = cate['id']
    index = ids_list.index(old_id)
    new_id = mapping_od_id_to_new_id[index]
    new_categories[i]['id'] = new_id

for i, ann in enumerate(new_annotations):
    old_id = ann['category_id']
    index = ids_list.index(old_id)
    new_id = mapping_od_id_to_new_id[index]
    new_annotations[i]['category_id'] = new_id



new_injson = {}
new_injson['info'] = info
new_injson['annotations'] = new_annotations
if TRAIN_PLUS_VAL == True:
    new_injson['images'] = images + images_val
else:
    new_injson['images'] = images
new_injson['licenses'] = license
new_injson['categories'] = new_categories
print('number of ann : ', len(new_annotations))
print(new_categories)

save_path = './new_json_file'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if TRAIN_PLUS_VAL == True:
    save_dir = os.path.join(save_path, 'new_train_plus_val_kitchen.json')
else:
    save_dir = os.path.join(save_path, 'new_train_single_kitchen.json')
with open(save_dir, 'w') as fp:
    json.dump(new_injson, fp)




