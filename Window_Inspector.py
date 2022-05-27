import os
import torch
from torchvision import models
import math
import numpy as np
import re
import cv2
import albumentations as A  # our data augmentation library
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2

from torchvision.utils import save_image
import shutil


# User parameters
SAVE_NAME_OD_1 = "./Models-OD/Window_Edge_Finder-OD-1000.model"
SAVE_NAME_OD_2 = "./Models-OD/Window-OD-615.model"
DATASET_PATH_1 = "./Training_Data/" + SAVE_NAME_OD_1.split("./Models-OD/",1)[1].split("-",1)[0] +"/"
IMAGE_SIZE              = int(re.findall(r'\d+', SAVE_NAME_OD_1)[-1] ) # Row and column number 
DATASET_PATH_2 = "./Training_Data/" + SAVE_NAME_OD_2.split("./Models-OD/",1)[1].split("-",1)[0] +"/"
TO_PREDICT_PATH         = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH          = "./Images/Prediction_Images/Predicted_Images/"
# PREDICTED_PATH        = "C:/Users/troya/.spyder-py3/ML-Defect_Detection/Images/Prediction_Images/To_Predict_Images/"
SAVE_ANNOTATED_IMAGES   = True
SAVE_ORIGINAL_IMAGE     = False
SAVE_CROPPED_IMAGES     = False
DIE_SPACING_SCALE       = 0.99
MIN_SCORE               = 0.8


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        full_path = os.path.join(dir, f)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)


# Creates class folder
def makeDir(dir, classes_2):
    for classIndex, className in enumerate(classes_2):
        os.makedirs(dir + className, exist_ok=True)


# Deletes unnecessary string in file name
def replaceFileName(slot_path):
    for file_name in os.listdir(slot_path):
        file_path = os.path.join(slot_path, file_name)
        # For loop with row number as "i" will take longer, so yes below seems
        #   redundant writing each number 1 by 1, but has to be done.
        os.rename(file_path, 
                  file_path.replace("Stitcher-Snaps_for_8in_Wafer_Pave.", "")\
                          .replace("Die-1_Pave.", "")\
                          .replace("Die1_Pave.", "")\
                          .replace("Med_El-A_River_1_Pave.", "")\
                          .replace("new_RefDes_1_PaveP1.", "")\
                          .replace("new_RefDes_1_Pave.", "")\
                          .replace("Window_Die1_Pave.", "")\
                          .replace("TPV2_Pave.", "")\
                          .replace("A-Unity_Pave.", "")\
                          .replace("Window_", "")\
                          .replace("Row_1.", "Row_01.")\
                          .replace("Col_1.", "Col_01.")\
                          .replace("Row_2.", "Row_02.")\
                          .replace("Col_2.", "Col_02.")\
                          .replace("Row_3.", "Row_03.")\
                          .replace("Col_3.", "Col_03.")\
                          .replace("Row_4.", "Row_04.")\
                          .replace("Col_4.", "Col_04.")\
                          .replace("Row_5.", "Row_05.")\
                          .replace("Col_5.", "Col_05.")\
                          .replace("Row_6.", "Row_06.")\
                          .replace("Col_6.", "Col_06.")\
                          .replace("Row_7.", "Row_07.")\
                          .replace("Col_7.", "Col_07.")\
                          .replace("Row_8.", "Row_08.")\
                          .replace("Col_8.", "Col_08.")\
                          .replace("Row_9.", "Row_09.")\
                          .replace("Col_9.", "Col_09.")\
                          .replace(".p0", "")\
                          .replace(".p1", "")\
                          .replace(".20",".P_")\
                          .replace(".21", "P_1")
                          )



# Starting stopwatch to see how long process takes
start_time = time.time()

# Deletes images already in "Predicted_Images" folder
deleteDirContents(PREDICTED_PATH)


dataset_path_1 = DATASET_PATH_1

#load classes
coco_1 = COCO(os.path.join(dataset_path_1, "train", "_annotations.coco.json"))
categories_1 = coco_1.cats
n_classes_1 = len(categories_1.keys())
categories_1

classes_1 = [i[1]['name'] for i in categories_1.items()]

# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features_1 = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features_1, n_classes_1)


# Loads inspection part
# ------------
dataset_path_2 = DATASET_PATH_2

#load classes
coco_2 = COCO(os.path.join(dataset_path_2, "train", "_annotations.coco.json"))
categories_2 = coco_2.cats
n_classes_2 = len(categories_2.keys())
categories_2

classes_2 = [i[1]['name'] for i in categories_2.items()]

# lets load the faster rcnn model
model_2 = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features_2 = model_2.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model_2.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features_2, n_classes_2)



# Loads last saved checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

if os.path.isfile(SAVE_NAME_OD_1) and os.path.isfile(SAVE_NAME_OD_2):
    checkpoint_1 = torch.load(SAVE_NAME_OD_1, map_location=map_location)
    model_1.load_state_dict(checkpoint_1)
    
    checkpoint_2 = torch.load(SAVE_NAME_OD_2, map_location=map_location)
    model_2.load_state_dict(checkpoint_2)

model_1 = model_1.to(device)
model_2 = model_2.to(device)

model_1.eval()
model_2.eval()
torch.cuda.empty_cache()

transforms_1 = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
    ToTensorV2()
])


replaceFileName(TO_PREDICT_PATH)

# Start FPS timer
fps_start_time = time.time()

color_list =['green', 'red', 'magenta', 'blue', 'orange', 'cyan', 'lime', 'turquoise', 'yellow']
ii = 0
for image_name in os.listdir(TO_PREDICT_PATH):
    image_path = os.path.join(TO_PREDICT_PATH, image_name)
    
    image_b4_color = cv2.imread(image_path)
    orig_image = image_b4_color
    image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)
    
    transformed_image = transforms_1(image=image)
    transformed_image = transformed_image["image"]
    
    if ii == 0:
        line_width = round(transformed_image.shape[1] * 0.00214)
    
    with torch.no_grad():
        prediction_1 = model_1([(transformed_image/255).to(device)])
        pred_1 = prediction_1[0]
    
    dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
    die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE].tolist()
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    die_scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE].tolist()
    
    if SAVE_ANNOTATED_IMAGES:
        predicted_image = draw_bounding_boxes(transformed_image,
            boxes = dieCoordinates,
            # labels = [classes_1[i] for i in die_class_indexes], 
            # labels = [str(round(i,2)) for i in die_scores], # SHOWS SCORE IN LABEL
            width = line_width,
            colors = 'magenta'
            )
        
        # ADDED FOR WINDOW
        # =========================================================================
        # Marks coordinates of edges of window dies
        horz_edge_found = False
        vert_edge_found = False
        for die_coordinate in dieCoordinates:
            # Finds horizontal edge
            if ( (die_coordinate[2] - die_coordinate[0]) > 500 
                and horz_edge_found == False):
                horz_edge_found = True
                horz_edge_x1 = die_coordinate[0]
                horz_edge_y1 = die_coordinate[1]
                horz_edge_x2 = die_coordinate[2]
                horz_edge_y2 = die_coordinate[3]
                if horz_edge_y1 < 500:
                    horz_edge_side = "Top"
                else:
                    horz_edge_side = "Bottom"
            # Finds verticle edge
            if ( (die_coordinate[3] - die_coordinate[1]) > 500 
                and vert_edge_found == False):
                vert_edge_found = True
                vert_edge_x1 = die_coordinate[0]
                vert_edge_y1 = die_coordinate[1]
                vert_edge_x2 = die_coordinate[2]
                vert_edge_y2 = die_coordinate[3]
                if vert_edge_x1 < 500:
                    vert_edge_side = "Left"
                else:
                    vert_edge_side = "Right"
        
        if horz_edge_found:
            if horz_edge_side == "Top":
                horz_edge_cutoff_y = horz_edge_y2 + 180
                predicted_image[:,
                                :int(horz_edge_cutoff_y)] = ( 
                    (predicted_image[:,
                                     :int(horz_edge_cutoff_y)]/2).type(torch.uint8) )
            else: 
                horz_edge_cutoff_y = horz_edge_y1 - 60
                predicted_image[:,
                                int(horz_edge_cutoff_y):] = ( 
                    (predicted_image[:,
                                     int(horz_edge_cutoff_y):]/2).type(torch.uint8) )
        if vert_edge_found:
            if not horz_edge_found:
                horz_edge_cutoff_y = 0
            
            if vert_edge_side == "Left":
                vert_edge_cutoff_x = vert_edge_x2 + 180
                if horz_edge_found and horz_edge_side == "Bottom":
                    predicted_image[:,
                                    :int(horz_edge_cutoff_y),
                                    :int(vert_edge_cutoff_x)] = ( 
                        (predicted_image[:,
                                         :int(horz_edge_cutoff_y),
                                         :int(vert_edge_cutoff_x)]/2).type(torch.uint8) )
                else:
                    predicted_image[:,
                                    int(horz_edge_cutoff_y):,
                                    :int(vert_edge_cutoff_x)] = ( 
                        (predicted_image[:,
                                         int(horz_edge_cutoff_y):,
                                         :int(vert_edge_cutoff_x)]/2).type(torch.uint8) )
            else: 
                vert_edge_cutoff_x = vert_edge_x1 - 520
                if horz_edge_found and horz_edge_side == "Bottom":
                    predicted_image[:,
                                    :int(horz_edge_cutoff_y),
                                    int(vert_edge_cutoff_x):] = ( 
                        (predicted_image[:,
                                         :int(horz_edge_cutoff_y),
                                         int(vert_edge_cutoff_x):]/2).type(torch.uint8) )
                else:
                    predicted_image[:,
                                    int(horz_edge_cutoff_y):,
                                    int(vert_edge_cutoff_x):] = ( 
                        (predicted_image[:,
                                         int(horz_edge_cutoff_y):,
                                         int(vert_edge_cutoff_x):]/2).type(torch.uint8) )
        
        # save_image((predicted_image/255), "test.jpg")
        
        # Inspection part
        # --------------------------------------------------------------------
        # What to predict
        if horz_edge_found:
            if horz_edge_side == "Top":
                horz_edge_cutoff_y = horz_edge_y2 + 180
                inspect_image = predicted_image[:, int(horz_edge_cutoff_y):]
            else:
                horz_edge_cutoff_y = horz_edge_y1 - 60
                inspect_image = predicted_image[:, :int(horz_edge_cutoff_y)]
        if vert_edge_found:
            if vert_edge_side == "Left":
                vert_edge_cutoff_x = vert_edge_x2 + 180
                if horz_edge_found:
                    inspect_image = inspect_image[:, :, int(vert_edge_cutoff_x):]
                else:
                    inspect_image = predicted_image[:, :, int(vert_edge_cutoff_x):]
            else:
                vert_edge_cutoff_x = vert_edge_x1 - 520
                if horz_edge_found:
                    inspect_image = inspect_image[:, :, :int(vert_edge_cutoff_x)]
                else:
                    inspect_image = predicted_image[:, :, :int(vert_edge_cutoff_x)]
        
        
        with torch.no_grad():
            prediction_2 = model_2([(inspect_image/255).to(device)])
            pred_2 = prediction_2[0]
        
        dieCoordinates_2 = pred_2['boxes'][pred_2['scores'] > MIN_SCORE]
        die_class_indexes_2 = pred_2['labels'][pred_2['scores'] > MIN_SCORE].tolist()
        # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
        die_scores_2 = pred_2['scores'][pred_2['scores'] > MIN_SCORE].tolist()
        
        inspect_image_bb = draw_bounding_boxes(inspect_image,
            boxes = dieCoordinates_2,
            # labels = [classes_2[i] for i in die_class_indexes_2], 
            # labels = [str(round(i,2)) for i in die_scores_2], # SHOWS SCORE IN LABEL
            width = line_width,
            colors = [color_list[i] for i in die_class_indexes_2]
            )
        # --------------------------------------------------------------------
        
        # Combining inspection section to main image
        if vert_edge_found:
            if vert_edge_side == "Left":
                vert_edge_cutoff_x = vert_edge_x2 + 180
                
                if horz_edge_found:
                    if horz_edge_side == "Top":
                        horz_edge_cutoff_y = horz_edge_y2 + 180
                    
                        predicted_image[:, 
                                        int(horz_edge_cutoff_y):, 
                                        int(vert_edge_cutoff_x):] = inspect_image_bb
                    else:
                        horz_edge_cutoff_y = horz_edge_y1 - 60
                    
                        predicted_image[:, 
                                        :int(horz_edge_cutoff_y), 
                                        int(vert_edge_cutoff_x):] = inspect_image_bb
                else:
                    predicted_image[:, 
                                    :, 
                                    int(vert_edge_cutoff_x):] = inspect_image_bb
            else:
                vert_edge_cutoff_x = vert_edge_x1 - 520
                
                if horz_edge_found:
                    if horz_edge_side == "Top":
                        horz_edge_cutoff_y = horz_edge_y2 + 180
                    
                        predicted_image[:, 
                                        int(horz_edge_cutoff_y):, 
                                        :int(vert_edge_cutoff_x)] = inspect_image_bb
                    else:
                        horz_edge_cutoff_y = horz_edge_y1 - 60
                    
                        predicted_image[:, 
                                        :int(horz_edge_cutoff_y), 
                                        :int(vert_edge_cutoff_x)] = inspect_image_bb
                else:
                    predicted_image[:, 
                                    :, 
                                    :int(vert_edge_cutoff_x)] = inspect_image_bb
        
        
        # =========================================================================
        
        # Saves full image with bounding boxes
        if len(die_class_indexes_2) != 0:
            save_image((predicted_image/255), PREDICTED_PATH + image_name)
        
        # save_image((predicted_image/255), PREDICTED_PATH + image_name)
        
    if SAVE_ORIGINAL_IMAGE and len(die_class_indexes) != 0:
        cv2.imwrite(PREDICTED_PATH + image_name + "Original.jpg", orig_image)
    

    if len(os.listdir(TO_PREDICT_PATH)) > 1000:
        tenScale = 1000
    else:
        tenScale = 100

    ii += 1
    if ii % tenScale == 0:
        fps_end_time = time.time()
        fps_time_lapsed = fps_end_time - fps_start_time
        print("  " + str(ii) + " of " 
              + str(len(os.listdir(TO_PREDICT_PATH))), 
              "-",  round(tenScale/fps_time_lapsed, 2), "FPS")
        fps_start_time = time.time()


print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)