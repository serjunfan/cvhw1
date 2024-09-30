
import os
import re
import cv2
import json
import itertools
import numpy as np
from glob import glob
import scipy.io as sio
from PIL import Image


MAX_N = 100

categories = [
    {"id": 0,"name": "Person"},
    {"id": 1,"name": "Head"},
    {"id": 2,"name": "Face"},
    {"id": 3,"name": "Glasses"},
    {"id": 4,"name": "Face-mask-medical"},
    {"id": 5,"name": "Face-guard"},
    {"id": 6,"name": "Ear"},
    {"id": 7,"name": "Earmuffs"},
    {"id": 8,"name": "Hands"},
    {"id": 9,"name": "Gloves"},
    {"id": 10,"name": "Foot"},
    {"id": 11,"name": "Shoes"},
    {"id": 12,"name": "Safety-vest"},
    {"id": 13,"name": "Tools"},
    {"id": 14,"name": "Helmet"},
    {"id": 15,"name": "Medical-suit"},
    {"id": 16,"name": "Safety-suit"},
]
train_processed = valid_processed = test_processed = 0

phases = ["train", "valid", "test"]
for phase in phases:
    label_dir = "datasets/{}/labels".format(phase)
    image_dir = "datasets/{}/images".format(phase)
    
    res_file = {
        "categories": categories,
        "images": [],
        "annotations": []
    }
    json_file = "{}.json".format(phase)
    
    annot_count = 0
    image_id = 0
    processed = 0
    if phase == "test":
        for filename in os.listdir(image_dir):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            img_w, img_h = img.size
            res_file["images"].append({
                "id": image_id,
                "file_name": filename,
                "width": img_w,
                "height": img_h,
            })
            processed += 1
            image_id += 1
        test_processed = processed
        break
        
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            image_extensions = ['.jpeg', '.jpg', '.png']
            image_file_name = None
            for ext in image_extensions:
                image_name = filename.replace('.txt', ext)
                if os.path.exists(os.path.join(image_dir, image_name)):
                    image_file_name = image_name
                    break
            if image_file_name is None:
                print(f"Warning: No image found for {filename}")
                continue
            
            #image_file_name = filename.replace('.txt', '.jpeg')  # Assuming .jpg images
            img_path = os.path.join(image_dir, image_file_name)
            img = Image.open(img_path)
            img_w, img_h = img.size
            res_file["images"].append({
                "id": image_id,
                "file_name": image_file_name,
                "width": img_w,
                "height": img_h,
            })
            with open(os.path.join(label_dir, filename), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    category_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
        
                    # Convert to COCO bounding box format (x_min, y_min, width, height)
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    res_file["annotations"].append({
                        "id": annot_count,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x_min, y_min, width, height],
                        "area": width * height
                    })
                    annot_count += 1
            processed += 1
            image_id += 1
        with open(json_file, "w") as f:
            json_str = json.dumps(res_file)
            f.write(json_str)
        #indent 4 or not?
        
        #with open(output_json, 'w') as json_file:
            #json.dump(coco_format, json_file, indent=4)
        print("Processed {} {} images...".format(processed, phase))
        if phase == "train":
            train_processed = processed
        else:
            valid_processed = processed
print('train_processed = {}, valid_processed = {}, test_processed = {}'.format(train_processed, valid_processed, test_processed))
print("Done.")
