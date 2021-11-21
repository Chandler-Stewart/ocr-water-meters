import json, pandas as pd

import os, os.path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

image_data = pd.read_csv("TlkWaterMeters\data.tsv", sep="\t")
image_data.location = image_data.location.str.replace(r"'", r'"')

def get_img_data(row_num):
    row = image_data.loc[row_num]
    filename = row[0]
    img_name = filename[:-4]
    img_coords = json.loads(row.iloc[2])
    img_coords.pop('type')
    img_coords['filename'] = filename
    return img_name, img_coords

def combine_json(num):
    full_json = {}
    for i in range(num):
        name, mask = get_img_data(i)
        full_json[name] = mask
    
    return full_json

data = combine_json(len(image_data))

with open("test.json", "w") as write_file:
    json.dump(data, write_file)