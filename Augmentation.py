import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
import torch
from torchvision import datasets, models
from torch.utils.data import DataLoader
import copy
import math
import re
import cv2
import albumentations as A  # our data augmentation library
# remove warnings (optional)
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm  # progress bar
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2


# User parameters
IMAGES_TO_AUGMENT_PATH  = "./Images/Prediction_Images/To_Predict/"
IMAGES_AUGMENTED_PATH   = "./Images/Prediction_Images/Predicted_Images/"
AUGMENTS_PER_IMAGE      = 10

# Transformation Parameters:
BLUR_PROB           = 0.15  # Default: 0.15
DOWNSCALE_PROB      = 0.20  # Default: 0.20
NOISE_PROB          = 0.50  # Default: 0.5
ISONOISE_PROB       = 0.50  # Default: 0.5
MOTION_BLUR_PROB    = 0.20  # Default: 0.20
ROTATION            = 30    # Default: 30
BRIGHTNESS_CHANGE   = 0.20  # Default: 0.20
CONTRAST_CHANGE     = 0.20  # Default: 0.20
SATURATION_CHANGE   = 0.20  # Default: 0.20
HUE_CHANGE          = 0.05  # Default: 0.20
HORIZ_FLIP_CHANCE   = 0.25  # Default: 0.25
VERT_FLIP_CHANCE    = 0.20  # Default: 0.20


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec)))


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        full_path = os.path.join(dir, f)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)


def main():
    transform = A.Compose([
        # A.Resize(IMAGE_SIZE, IMAGE_SIZE), # I don't include anymore because OD models doesn't discriminate against size
        # A.Rotate(limit=[90,90], always_apply=True),
        A.GaussianBlur(blur_limit=(3, 5), p=BLUR_PROB),
        A.Downscale(scale_min=0.40, scale_max=0.99, p=DOWNSCALE_PROB),
        A.GaussNoise(var_limit=(1.0, 500.0), p=NOISE_PROB),
        A.ISONoise (color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=ISONOISE_PROB),
        A.MotionBlur(5, p=MOTION_BLUR_PROB),
        A.ColorJitter(brightness=BRIGHTNESS_CHANGE,
                      contrast=CONTRAST_CHANGE,
                      saturation=SATURATION_CHANGE,
                      hue=HUE_CHANGE,
                      p=0.40),
        A.HorizontalFlip(p=HORIZ_FLIP_CHANCE),
        A.VerticalFlip(p=VERT_FLIP_CHANCE),
        # A.RandomRotate90(p=0.2),
        A.Rotate(limit=[-ROTATION, ROTATION])
    ])


    for image_name in os.listdir(IMAGES_TO_AUGMENT_PATH):
        image_path = os.path.join(IMAGES_TO_AUGMENT_PATH, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for index in range(AUGMENTS_PER_IMAGE):

            # Apply the augmentation pipeline
            augmented_image = transform(image=image)["image"]
            # print(augmented_image)

            # Convert the image back to BGR format
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

            # Save the augmented image
            output_file = os.path.join(IMAGES_AUGMENTED_PATH, f"{image_name}___{index}.jpg")
            print(output_file)
            cv2.imwrite(output_file, augmented_image)





if __name__ == "__main__":
    # Starting stopwatch to see how long process takes
    start_time = time.time()

    # Deletes images already in "Predicted_Images" folder
    deleteDirContents(IMAGES_AUGMENTED_PATH)


    main()


    print("Done!")

    # Stopping stopwatch to see how long process takes
    end_time = time.time()
    time_lapsed = end_time - start_time
    time_convert(time_lapsed)







