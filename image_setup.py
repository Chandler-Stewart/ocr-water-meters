import math, os, os.path, glob, json, cv2, pandas as pd
import cv2

image_data = pd.read_csv('data_after_eda.csv')

def landscape(row):
    if row['height'] <= row['width']:
        return 1
    else:
        return 0

image_data['landscape'] = image_data.apply(landscape, axis=1)
print(image_data.head(3))

# while testing, this will remove the previously created images to restart the process
for dir in ['collage', 'images', 'masks']:
    temp_path = os.path.join('temp/pad-crop/', dir)
    files = glob.glob(temp_path + '/*.jpg')
    for f in files:
        os.remove(f)

def write_landscape():
    for dir in ['collage', 'images', 'masks']:
        for _, row in image_data.iterrows():
            img_path = os.path.join('TlkWaterMeters/', dir, row['photo_name'])
            temp_path = os.path.join('temp/pad-crop/', dir, row['photo_name'])
            img = cv2.imread(img_path)
            if row['landscape']:
                cv2.imwrite(temp_path, img)
            else:
                rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(temp_path, rotated)

def rotate_coords():
    for index, row in image_data.iterrows():
        if row['landscape'] == 0:
            for coords in ['location', 'pixel_coords']:
                row[coords] = row[coords].replace("\'", "\"")
                json_coords = json.loads(row[coords])
                replacement = []
                for coord in json_coords:
                    replacement_coord = {}
                    if coords == 'location':
                        replacement_coord['x'], replacement_coord['y'] = coord['y'], 1 - coord['x']
                    else:
                        replacement_coord['x'], replacement_coord['y'] = coord['y'], row['width'] - coord['x']
                    replacement.append(replacement_coord)

                data = {'coordinates': replacement}

                image_data.loc[index, coords] = json.dumps(data)

            image_data.loc[index, 'width'], image_data.loc[index, 'height'] = row['height'], row['width']
            
        else:
            for coords in ['location', 'pixel_coords']:
                row[coords] = row[coords].replace("\'", "\"")
                json_coords = json.loads(row[coords])
                data = {'coordinates': json_coords}
                image_data.loc[index, coords] = json.dumps(data)

def pad(src, pxl_diff):
    left_pad = pxl_diff - (pxl_diff//2)
    right_pad = pxl_diff - left_pad
    img_padded = cv2.copyMakeBorder(src, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, [255, 255, 255])
    return img_padded

def crop(src, pxl_diff):
    height, width = src.shape[:2]
    left_crop = pxl_diff - (pxl_diff//2)
    right_crop = pxl_diff - left_crop
    img_cropped = src[0:height, left_crop:width-right_crop]
    return img_cropped

def scale(src, height, width):
    scale_percent = width / 1333 # percent of original size
    width = 1333
    height = int(height * scale_percent)
    print(f'rescaled to {width} x {height}')
    # resize image
    resized = cv2.resize(src, (width, height), interpolation = cv2.INTER_AREA)
    return resized

def equalize():
    for _, row in image_data.iterrows():
        for dir in ['collage', 'images', 'masks']:
            temp_path = os.path.join('temp/pad-crop/', dir, row['photo_name'])
            img = cv2.imread(temp_path)
            changed_flag = 0
            height, width = row['height'], row['width']
            if width != 1333:
                img = scale(img, height, width)
                changed_flag = 1
            '''if height > 1000:
                img = crop(img, abs(height - 1000))
                changed_flag = 1
            elif height < 1000:
                img = pad(img, abs(height - 1000))
                changed_flag = 1'''
            if changed_flag == 1:
                cv2.imwrite(temp_path, img)


write_landscape()
rotate_coords()
print(image_data.head(3))
equalize()


#image_data['pixel_coords'] = image_data.apply(rotate_img_coords, axis=1)
#image_data['location'] = image_data.apply(rotate_img_abs_coords, axis=1)
#print(image_data.head(3))

#top, bot = pad(0, 5)
#print(top)
#print(bot)