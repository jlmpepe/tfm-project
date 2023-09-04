"""This module allows to generate the ground truth table from the COCO dataset"""
import shutil
import numpy as np
import pandas as pd
import os
from pycocotools.coco import COCO

coco = COCO('instances_train2017.json')
SRC_FOLDER_NAME = 'images2017'
DST_FOLDER_NAME = 'dataset'

# Classes of images to be included in the datasets
accepted_supercategories = ['person', 'animal']

mapNewNamesOldNames = {
    'motorcycle': 'motorbike',
    'airplane': 'aeroplane',
    'couch': 'sofa',
    'potted plant': 'pottedplant',
    'dining table': 'diningtable',
    'tv': 'tvmonitor'
}

def save_dataset(data, filename):
    """Export the ground truth table to a CSV file"""
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(filename, sep='\t', encoding='utf-8', index=False)

def delete_zeros(annotation):
    """Transforms zeros into ones"""
    new_annotation = annotation
    # Comment on this line if datasets are being generated for YOLO
    # new_annotation = np.round(new_annotation)
    return map(lambda element: 1 if element == 0 else element, new_annotation)

def copy_image(src, dst):
    """Copy the image found in the src path and paste it into the dst path"""
    if not os.path.exists(dst):
        shutil.copy(src, dst)

def filter_categories():
    """Returns an array with the accepted categories and a dictionary
    whose key is the id of the category and value is the category"""
    categories = coco.loadCats(coco.getCatIds())
    categories_id_cat = {}
    accepted_category_ids = []
    for category in categories:
        modified_category = category
        if modified_category['name'] in mapNewNamesOldNames:
            modified_category['name'] = mapNewNamesOldNames[modified_category['name']]
        categories_id_cat[category['id']] = modified_category
        if category['supercategory'] in accepted_supercategories:
            print(f'ID: {category["id"]} Name: {category["name"]}')
            accepted_category_ids.append(category['id'])
    return [accepted_category_ids, categories_id_cat]

# Variable to prevent an image from being in two different datasets
all_image_ids = []
def create_dataset(num_images, accepted_category_ids, categories_id_cat):
    """Generates ground truth table"""
    dataset = {"path": []}
    index = 0

    for category_id in accepted_category_ids:
        category = categories_id_cat[category_id]
        dataset[category['name']] = ["[]"] * num_images * len(accepted_category_ids)

    for category_id in accepted_category_ids:
        img_ids = coco.getImgIds(catIds=[category_id])
        img_ids_clean = []
        for img_id in img_ids:
            if len(img_ids_clean) == num_images:
                break
            if not img_id in all_image_ids:
                img_ids_clean.append(img_id)
        images = coco.loadImgs(img_ids_clean)

        for image in images:
            image_id = image['id']
            all_image_ids.append(image_id)
            copy_image(f"{SRC_FOLDER_NAME}/{image['file_name']}", f"{DST_FOLDER_NAME}/{image['file_name']}")
            image_annotations = coco.getAnnIds(imgIds=image_id)
            annotations = coco.loadAnns(image_annotations)

            new_annotations = {}
            for annotation in annotations:
                category = categories_id_cat[annotation['category_id']]

                if category['id'] in accepted_category_ids:
                    category_name = category['name']

                    annotation_int = delete_zeros(annotation['bbox'])
                    if category_name in new_annotations:
                        bbox = new_annotations[category_name]
                        new_annotations[category_name] = bbox[:len(bbox)-1]+f";{','.join(map(str, annotation_int))}]"
                    else:
                        new_annotations[category_name] = f"[{','.join(map(str, annotation_int))}]"

            if new_annotations:
                dataset["path"].append(f"{DST_FOLDER_NAME}/{image_id:012d}.jpg")
                for key, value in new_annotations.items():
                    dataset[key][index] = value
                index += 1
    return dataset

[accepted_categories, map_category_id_cat] = filter_categories()
print(f'Total number of classes: {len(accepted_categories)}')
NUM_IMAGES_PER_CATEGORY = 48
train_dataset = create_dataset(
    NUM_IMAGES_PER_CATEGORY,
    accepted_categories,
    map_category_id_cat
)
val_dataset = create_dataset(
    int(0.2 * NUM_IMAGES_PER_CATEGORY),
    accepted_categories,
    map_category_id_cat
)
test_dataset = create_dataset(
    int(0.1 * NUM_IMAGES_PER_CATEGORY),
    accepted_categories,
    map_category_id_cat
)

train_dataframe = {
    **train_dataset
}

val_dataframe = {
    **val_dataset
}

test_dataframe = {
    **test_dataset
}

save_dataset(train_dataframe, 'train.csv')
save_dataset(val_dataframe, 'val.csv')
save_dataset(test_dataframe, 'test.csv')
