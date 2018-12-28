import json
import os
import pandas as pd
import datetime
import cv2
phase="train"
ROOT_DIR = 'train'
IMAGE_DIR = os.path.join(ROOT_DIR, "shapes_train2018")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

INFO = {
    "description": "Deeplesion Dataset",
    "url": "",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "wcw",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Deeplesion",
        "url": ""
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': '1',
        'supercategory': 'lesion',
    },
    {
        'id': 2,
        'name': '2',
        'supercategory': 'lesion',
    },
    {
        'id': 3,
        'name': '3',
        'supercategory': 'lesion',
    },
    {
        'id':4 ,
        'name': '4',
        'supercategory': 'lesion',
    },
    {
        'id': 5,
        'name': '5',
        'supercategory': 'shape',
    },
    {
        'id': 6,
        'name': '6',
        'supercategory': 'lesion',
    },
    {
        'id': 7,
        'name': '7',
        'supercategory': 'lesion',
    },
    {
        'id': 8,
        'name': '8',
        'supercategory': 'lesion',
    },
]
dataset = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
}
root_path = 'C:\\Users\\officer\\Desktop\\data\\coco'
path="F:\\Data\\Deeplesion\\Key_slices\\key_show\\"
data=pd.read_csv("F:\\Data\\Deeplesion\\DL_info.csv")
list=data.values.tolist()
count=0;
for i in range(len(list)):
#for i in range(5):
    op = os.path.exists(path + str(i + 1) + ".png")
    type = str(list[i][9])

    if op == True and type!="-1":
        idname = str(i + 1)+".png"
        name = list[i][0]
        pos = list[i][6].split(',')
        x1=round(float(pos[0]),1)
        y1=round(float(pos[1]),1)
        x2=round(float(pos[2]),1)
        y2=round(float(pos[3]),1)
        width=x2-x1
        height=y2-y1
        dataset['images'].append({'file_name': idname,
                                      'id': i,
                                      'width': 512,
                                      'height': 512})
        dataset['annotations'].append({
            'area': 262144,
            'bbox': [x1, y1, round(width,1), round(height,1)],
            'category_id': int(type),
            'id': i,
            'image_id': i,
            'iscrowd': 0,
            # mask, 矩形是从左上角点按顺时针的四个顶点
            'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
           })
        print(i)
        count=count+1

folder = os.path.join(root_path, 'annotations')
if not os.path.exists(folder):
  os.makedirs(folder)
json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
with open(json_name, 'w') as f:
  json.dump(dataset, f)
print(count)