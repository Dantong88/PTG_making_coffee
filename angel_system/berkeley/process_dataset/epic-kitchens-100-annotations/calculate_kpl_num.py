import os

path = './infer_list'

person_ids = os.listdir(path)
j = 0
for id in range(37):
    if id + 1 > 9:
        person_id = 'P' + str(id + 1)
    else:
        person_id = 'P0' + str(id + 1)
    path_ = os.path.join(path, person_id)
    pkl_lists = os.listdir(path_)
    i = 0
    for pkl in pkl_lists:
        if pkl[-3:] == 'txt':
            i = i + 1
    j = j + 1
    print(person_id + ' : ' + str(i))
