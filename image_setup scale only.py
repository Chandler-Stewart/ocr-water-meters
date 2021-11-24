import math, os, os.path, glob, json, cv2, pandas as pd
import numpy as np

from shapely.geometry import Polygon, MultiPolygon

import cv2

from sklearn.model_selection import train_test_split

image_data = pd.read_csv('data_after_eda.csv')

train, val = train_test_split(image_data, random_state = 12345, shuffle=True)
df_split = {"train": train, "val": val}

# while testing, this will remove the previously created images to restart the process
for dir1 in list(df_split.keys()):
    for dir2 in ['collage', 'images', 'masks']:
        temp_path = os.path.join('temp/pad-crop/', dir1, dir2)
        files = glob.glob(temp_path + '/*.jpg')
        for f in files:
            os.remove(f)

#https://www.immersivelimit.com/create-coco-annotations-from-scratch
def get_annotation(df, i):
    json_coords = json.loads((df.loc[i,'location']).replace("\'", "\""))
    poly_coords = []
    for point in json_coords:
        poly_coords.append((point['x']*1000, point['y']*1000))
    poly = Polygon(poly_coords)

    segmentations = []
    segmentation = np.array(poly.exterior.coords).ravel().tolist()
    segmentations.append(segmentation)

    #multi_poly = MultiPolygon(poly)
    x, y, max_x, max_y = poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': 0,
        'image_id': (image_data.loc[i,'photo_name']).split('_')[1],
        'category_id': 1,
        'id': (image_data.loc[i,'photo_name']).split('_')[1],
        'bbox': bbox,
        'area': area
    }

    return annotation

def get_image(df, i):
    file_name = df.loc[i,'photo_name']
    
    img = {
        "license": 4,
        "file_name": file_name,
        "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "height": 1000,
        "width": 1000,
        "date_captured": "2013-11-14 17:02:52",
        "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        "id": (image_data.loc[i,'photo_name']).split('_')[1]
    }

    return img

def scale(src):
    resized = cv2.resize(src, (1000, 1000), interpolation = cv2.INTER_AREA)
    return resized

def scale_all(df, dir1):
    for _, row in df.iterrows():
        for dir2 in ['collage', 'images', 'masks']:
            read_path = os.path.join('TlkWaterMeters/', dir2, row['photo_name'])
            temp_path = os.path.join('temp/pad-crop/', dir1, dir2, row['photo_name'])
            img = cv2.imread(read_path)
            height, width = row['height'], row['width']
            if width != 1000 or height !=1000:
                img = scale(img)
            cv2.imwrite(temp_path, img)

def to_coco_json(df, dir1):
    annotations = []
    images = []
    for i in list(df.index.values):
        annotation = get_annotation(df, i)
        image = get_image(df, i)
        annotations.append(annotation)
        images.append(image)
    
    coco_json = {
        "info": {
            "description": "COCO formatted json for image segmentation",
            "contributer": "Chandler Stewart",
            "year": 2021
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by/2.0/",
                "id": 4,
                "name": "Attribution License"
            }
        ],
        "images": images,
        "annotations": annotations,
        "categories": [{"supercategory": "meter","id": 1,"name": "read-out"}]
    }

    read_path = os.path.join('temp/pad-crop/', dir1)
    with open(read_path + dir1 + '.json', 'w') as f:
        json.dump(coco_json, f)
    return coco_json

for each in df_split:
    scale_all(df_split[each], each)
    to_coco_json(df_split[each], each)

