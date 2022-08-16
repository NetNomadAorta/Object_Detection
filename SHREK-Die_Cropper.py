import os
import torch
from torchvision import models
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
import xlsxwriter


# User parameters
SAVE_NAME_OD_1 = "./Models-OD/SHREK_Die_Cropper-0.model"
DATASET_PATH_1 = "./Training_Data/" + SAVE_NAME_OD_1.split("./Models-OD/",1)[1].split("-",1)[0] +"/"
SAVE_NAME_OD_2 = "./Models-OD/Lord_of_Models-0.model"
DATASET_PATH_2 = "./Training_Data/" + SAVE_NAME_OD_2.split("./Models-OD/",1)[1].split("-",1)[0] +"/"
TO_PREDICT_PATH         = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH          = "./Images/Prediction_Images/Predicted_Images/"
MIN_SCORE_1             = 0.6 # Default 0.5
MIN_SCORE_2             = 0.6


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



# Main()
# =============================================================================

# Starting stopwatch to see how long process takes
start_time = time.time()

# Deletes images already in "Predicted_Images" folder
deleteDirContents(PREDICTED_PATH)

dataset_path_1 = DATASET_PATH_1

#load classes
coco_1 = COCO(os.path.join(dataset_path_1, "train", "_annotations.coco.json"))
categories_1 = coco_1.cats
n_classes_1 = len(categories_1.keys())

classes_1 = [i[1]['name'] for i in categories_1.items()]

# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features_1 = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features_1, n_classes_1)


# Loads inspection part
# ------------
dataset_path_2 = DATASET_PATH_2

#load classes
coco_2 = COCO(os.path.join(dataset_path_2, "train", "_annotations.coco.json"))
categories_2 = coco_2.cats
n_classes_2 = len(categories_2.keys())

classes_2 = [i[1]['name'] for i in categories_2.items()]

# lets load the faster rcnn model
model_2 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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
    A.Rotate(limit=[-90,-90], always_apply=True),
    ToTensorV2()
])

# Creates a workbook
workbook = xlsxwriter.Workbook(PREDICTED_PATH + 'Bump_Diameter_Measurements.xlsx')
# Create worksheet
worksheet = workbook.add_worksheet("Report")
# Adds headers in worksheet
worksheet.write(1-1, 1-1, 
                "Row #"
                )
worksheet.write(1-1, 2-1, 
                "Col #"
                )
worksheet.write(1-1, 3-1, 
                "Bump #"
                )
worksheet.write(1-1, 4-1, 
                "Diameter (μm)"
                )
red_background = workbook.add_format(
    {'bold': True,
     'font_color': 'white',
     'bg_color': 'red'
     }
    )

# Sets current Excel Row starting at
excel_row = 1

# Start FPS timer
fps_start_time = time.time()

color_list =['green', 'red', 'blue', 'magenta', 'orange', 'cyan', 'lime', 'turquoise', 'yellow']
pred_dict = {}
ii = 0
for image_name in os.listdir(TO_PREDICT_PATH):
    image_path = os.path.join(TO_PREDICT_PATH, image_name)
    
    image_b4_color = cv2.imread(image_path)
    orig_image = image_b4_color
    image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)
    
    transformed_image = transforms_1(image=image)
    transformed_image = transformed_image["image"]
    
    if ii == 0:
        line_width = max(round(transformed_image.shape[1] * 0.002), 1)
    
    with torch.no_grad():
        prediction_1 = model_1([(transformed_image/255).to(device)])
        pred_1 = prediction_1[0]
    
    dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE_1]
    die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE_1]
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    die_scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE_1]
    
    # Grabs row and column number from image name and corrects them
    path_row_number = int( re.findall(r'\d+', image_name)[0] )
    path_col_number = int( re.findall(r'\d+', image_name)[1] )
    path_part_number = int( re.findall(r'\d+', image_name)[2] )
    
    real_rowNum = 56 - path_row_number
    real_colNum = 35 - path_col_number
    
    if real_colNum < 10:
        real_colNum = "0" + str(real_colNum)
    
    if real_rowNum < 10:
        real_rowNum = "0" + str(real_rowNum)
        
    xmin = int(dieCoordinates[0][0])
    ymin = int(dieCoordinates[0][1])
    xmax = int(dieCoordinates[0][2])
    ymax = int(dieCoordinates[0][3])
    
    inspect_image = transformed_image[:, ymin:ymax, xmin:xmax]
    
    real_image_name = "Row_{}.Col_{}.P_0{}.jpg".format(real_rowNum, real_colNum, path_part_number)
    # save_image(inspect_image/255, 
    #             PREDICTED_PATH + real_image_name)
    
    with torch.no_grad():
        prediction_2 = model_2([(inspect_image/255).to(device)])
        pred_2 = prediction_2[0]
    
    dieCoordinates_2 = pred_2['boxes'][pred_2['scores'] > MIN_SCORE_2]
    die_class_indexes_2 = pred_2['labels'][pred_2['scores'] > MIN_SCORE_2]
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    die_scores_2 = pred_2['scores'][pred_2['scores'] > MIN_SCORE_2]
    
    # Labels each bump
    labels_found = []
    if path_part_number == 1:
        for dieCoordinate_index, dieCoordinate in enumerate(dieCoordinates_2):
            # Non Bumps
            if die_class_indexes_2[dieCoordinate_index] == 3:
                # labels_found.append(classes_2[die_class_indexes_2[dieCoordinate_index]])
                labels_found.append(" ")
            # Bottom bumps
            elif dieCoordinate[1] > 1700:
                bump_number = int( (dieCoordinate[0] - 40)/225 + 7 )
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # Top Bumps
            elif dieCoordinate[1] < 100:
                bump_number = 26 - int( (dieCoordinate[0] - 40)/225 + 1 )
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # Right Bumps
            elif dieCoordinate[0] > 1350:
                bump_number = 20 - int( (dieCoordinate[1] - 160)/225 + 1 )
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # Bump 39
            elif (dieCoordinate[0] > 550 and dieCoordinate[2] < 900
                  and dieCoordinate[1] > 1400 and dieCoordinate[3] < 1800):
                bump_number = 39
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # Bump 40
            elif (dieCoordinate[0] > 550 and dieCoordinate[2] < 900
                  and dieCoordinate[1] > 1200 and dieCoordinate[3] < 1500):
                bump_number = 40
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # Bump 41
            elif (dieCoordinate[0] > 200 and dieCoordinate[2] < 500
                  and dieCoordinate[1] > 1400 and dieCoordinate[3] < 1800):
                bump_number = 41
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # Bump 42
            elif (dieCoordinate[0] > 1100 and dieCoordinate[2] < 1450
                  and dieCoordinate[1] > 650 and dieCoordinate[3] < 950):
                bump_number = 42
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # Bump 43
            elif (dieCoordinate[0] > 1100 and dieCoordinate[2] < 1450
                  and dieCoordinate[1] > 400 and dieCoordinate[3] < 750):
                bump_number = 43
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # Bump 44
            elif (dieCoordinate[0] > 450 and dieCoordinate[2] < 750
                  and dieCoordinate[1] > 150 and dieCoordinate[3] < 450):
                bump_number = 44
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # For non-bumps
            else:
                labels_found.append("Unknown_Bump_#")
    elif path_part_number == 2:
        for dieCoordinate_index, dieCoordinate in enumerate(dieCoordinates_2):
            # Non Bumps
            if die_class_indexes_2[dieCoordinate_index] == 3:
                # labels_found.append(classes_2[die_class_indexes_2[dieCoordinate_index]])
                labels_found.append(" ")
            # Bottom bumps
            elif dieCoordinate[1] > 1700:
                bump_number = int( (dieCoordinate[0] - 190)/225 + 1 )
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # Top Bumps
            elif dieCoordinate[1] < 100:
                bump_number = 32 - int( (dieCoordinate[0] - 190)/225 + 1 )
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # Left Bumps
            elif dieCoordinate[0] < 100:
                bump_number = int( (dieCoordinate[1] - 160)/225 + 32 )
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # Bump 45
            elif (dieCoordinate[0] > 800 and dieCoordinate[2] < 1200
                  and dieCoordinate[1] > 200 and dieCoordinate[3] < 500):
                bump_number = 45
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # Bump 46
            elif (dieCoordinate[0] > 100 and dieCoordinate[2] < 500
                  and dieCoordinate[1] > 100 and dieCoordinate[3] < 500):
                bump_number = 46
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # Bump 47
            elif (dieCoordinate[0] > 1000 and dieCoordinate[2] < 1400
                  and dieCoordinate[1] > 1200 and dieCoordinate[3] < 1600):
                bump_number = 47
                bump_number = str(bump_number)
                labels_found.append(bump_number)
            # For non-bumps
            else:
                labels_found.append("Unknown_Bump_#")
    
    # labels_found = [str(index+1) for index, dieCoordinate in enumerate(dieCoordinates_2)]
    
    # Gets diameter size
    diameters_list = []
    for dieCoordinate_index, dieCoordinate in enumerate(dieCoordinates_2):
        if die_class_indexes_2[dieCoordinate_index] == 3:
            diameters_list.append(0)
            continue
        diameter_x = dieCoordinate[2] - dieCoordinate[0]
        diameter_y = dieCoordinate[3] - dieCoordinate[1]
        diameters_list.append( round(int( max(diameter_x, diameter_y) )*1.75) ) # 1.75 is the micron/pixel scale!
    
    
    # Writes data in Excel sheet
    for label_index, label in enumerate(labels_found):
        if "Unknown_Bump_#" in label or " " in label:
            continue
        
        # Adds to worksheet
        worksheet.write(excel_row, 1-1, 
                        int(real_rowNum)
                        )
        worksheet.write(excel_row, 2-1, 
                        int(real_colNum)
                        )
        worksheet.write(excel_row, 3-1, 
                        int(label)
                        )
        
        worksheet.write_url(excel_row, 4-1, 
                            ("C:/Users/troya/.spyder-py3/Object_Detection/" 
                             + PREDICTED_PATH + real_image_name
                             )
                            )
        
        if diameters_list[label_index] > 220:
            worksheet.write(excel_row, 4-1, 
                            diameters_list[label_index],
                            red_background
                            )
        else:
            worksheet.write(excel_row, 4-1, 
                            diameters_list[label_index]
                            )
        
        excel_row += 1
    
    
    inspect_image_bb = draw_bounding_boxes(inspect_image,
        boxes = dieCoordinates_2,
        labels = labels_found, 
        # labels = [str(round(i,2)) for i in die_scores_2], # SHOWS SCORE IN LABEL
        width = 1,
        colors = [color_list[i] for i in die_class_indexes_2],
        font = "arial.ttf",
        font_size = 20
        )
    
    save_image(inspect_image_bb/255, 
                PREDICTED_PATH + real_image_name)
    
    if len(os.listdir(TO_PREDICT_PATH)) > 2000:
        tenScale = 1000
    elif len(os.listdir(TO_PREDICT_PATH)) > 1000:
        tenScale = 500
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

# Sets width of columns
worksheet.set_column(0, 0, width=5.5)
worksheet.set_column(1, 1, width=5.5)
worksheet.set_column(2, 2, width=7)
worksheet.set_column(3, 3, width=12.5)

workbook.close()

print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)

# =============================================================================