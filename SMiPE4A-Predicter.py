import os
import sys
import torch
from torchvision import models
import math
from math import sqrt
import re
import cv2
import glob
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
SAVE_NAME_OD_1 = "./Models-OD/SMiPE4_Multi_Label-1090.model"
DATASET_PATH_1 = "./Training_Data/" + SAVE_NAME_OD_1.split("./Models-OD/",1)[1].split("-",1)[0] +"/"
IMAGE_SIZE     = int(re.findall(r'\d+', SAVE_NAME_OD_1)[-1] ) # Row and column number 
TO_PREDICT_PATH         = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH          = "./Images/Prediction_Images/Predicted_Images/"
MIN_SCORE_1             = 0.55 # Default 0.5
RENAME_TOGGLE           = True


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


# Deletes unnecessary string in file name
def replaceFileName(slot_path):
    for filename in glob.glob(slot_path + "/*"):
        # For loop with row number as "i" will take longer, so yes below seems
        #   redundant writing each number 1 by 1, but has to be done.
        os.rename(filename, 
                  filename.replace("Stitcher-Snaps_for_8in_Wafer_Pave.", "")\
                          .replace("Die-1_Pave.", "")\
                          .replace("Die1_Pave.", "")\
                          .replace("Med_El-A_River_1_Pave.", "")\
                          .replace("new_RefDes_1_PaveP1.", "")\
                          .replace("new_RefDes_1_Pave.", "")\
                          .replace("Window_Die1_Pave.", "")\
                          .replace("TPV2_Pave.", "")\
                          .replace("A-Unity_Pave.", "")\
                          .replace("E-Merlin_Pave.", "")\
                          .replace("A-Fang_Pave.", "")\
                          .replace("A-Soarin_Pave.", "")\
                          .replace("A-B4001_Pave.", "")\
                          .replace("A-RDSensor1_Pave.", "")\
                          .replace("A-Vapor_ROC3_Pave.", "")\
                          .replace("SMiPE4A_Pave.", "")\
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

# Loads inspection part
# ------------
dataset_path_1 = DATASET_PATH_1

#load classes
coco_1 = COCO(os.path.join(dataset_path_1, "train", "_annotations.coco.json"))
categories_1 = coco_1.cats
n_classes_1 = len(categories_1.keys())

classes_1 = [i[1]['name'] for i in categories_1.items()]

# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=500)
in_features_1 = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features_1, n_classes_1)



# Loads last saved checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

if os.path.isfile(SAVE_NAME_OD_1):
    checkpoint_1 = torch.load(SAVE_NAME_OD_1, map_location=map_location)
    model_1.load_state_dict(checkpoint_1)

model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

transforms_1 = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Rotate(limit=[90,90], always_apply=True),
    ToTensorV2()
])

# Creates a workbook
workbook = xlsxwriter.Workbook(PREDICTED_PATH + '0-Report.xlsx')

worksheet_list = []
for sheet_index in range(4):
    worksheet_list.append(workbook.add_worksheet(str(sheet_index)))

# Chooses each font and background color for the Excel sheet
font_color_list = ['white', 'black', 'white', 'white', 'black', 
                   'white', 'white', 'black', 'white']
bg_color_list = ['black', 'lime', 'red', 'green', 'yellow', 
                 'blue', 'magenta', 'cyan', 'gray']


# Chooses which font and background associated with each class
bin_colors_list = []
bin_bold_colors_list = []
for class_index in range(len(classes_1)):
    bin_colors_list.append(workbook.add_format(
        {'font_color': font_color_list[class_index],
         'bg_color': bg_color_list[class_index]}))
    bin_bold_colors_list.append(workbook.add_format(
        {'bold': True,
         'font_color': font_color_list[class_index],
         'bg_color': bg_color_list[class_index]}))
    
# For the "Not Tested Count" gray class
bin_colors_list.append(workbook.add_format(
    {'font_color': font_color_list[-1],
     'bg_color': bg_color_list[-1]}))
bin_bold_colors_list.append(workbook.add_format(
    {'bold': True,
     'font_color': font_color_list[-1],
     'bg_color': bg_color_list[-1]}))

# Removes unnecessary naming from original images
if RENAME_TOGGLE:
    print("  Renaming started.")
    replaceFileName(TO_PREDICT_PATH)
    print("  Renaming completed.")

# Start FPS timer
fps_start_time = time.time()

color_list =['white', 'gray', 'lime', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'gray']
all_dieNames = []
all_dieBinNumbers = []
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
    
    predicted_image = draw_bounding_boxes(transformed_image,
        boxes = dieCoordinates,
        # labels = [classes_1[i] for i in die_class_indexes], 
        # labels = [str(round(i,2)) for i in die_scores], # SHOWS SCORE IN LABEL
        width = line_width,
        colors = [color_list[i] for i in die_class_indexes],
        font = "arial.ttf",
        font_size = 20
        )
    
    # Grabs row and column number from image name and corrects them
    path_row_number = int( re.findall(r'\d+', image_name)[0] )
    path_col_number = int( re.findall(r'\d+', image_name)[1] )
    path_part_number = int( re.findall(r'\d+', image_name)[2] )
    
    if path_part_number % 8 == 0:
        part_row_number = 0
    else:
        part_row_number = (8 - path_part_number % 8) * 10
    part_col_number = ((path_part_number-1) // 8) * 10
    
    # Sets the box width and height
    if len(dieCoordinates) > 0:
        # If cutout boxes are at the beginning, this makes sure that it 
        #  finds a box in middle with full box width and height
        for box_index in range(len(dieCoordinates)):
            if (int(dieCoordinates[box_index][0]) > 150 
                and int(dieCoordinates[box_index][1]) > 150
                and int(dieCoordinates[box_index][2]) < (transformed_image.shape[2]-150)
                and int(dieCoordinates[box_index][3]) < (transformed_image.shape[1]-150)
                ):
                box_width = int(dieCoordinates[0][2]-dieCoordinates[0][0])
                box_height = int(dieCoordinates[0][3]-dieCoordinates[0][1])
                break
    
    # Sets spacing between dies
    die_spacing_max_width = int(box_width * .15) # I guessed
    die_spacing_width = 1 + round( (die_spacing_max_width/box_width)*0.80, 3)
    die_spacing_max_height = int(box_width * .15) # I guessed
    die_spacing_height = 1 + round( (die_spacing_max_height/box_width)*0.80, 3)
    
    # Grabbing max and min x and y coordinate values
    if len(dieCoordinates) > 0:
        
        # Because first boxes on edges can be cutoff, we do this
        min_x = torch.min(dieCoordinates[:, 0])
        min_y = torch.min(dieCoordinates[:, 1])
        max_x = torch.max(dieCoordinates[:, 2])
        max_y = torch.max(dieCoordinates[:, 3])
        
        for dieCoordinate_index, dieCoordinate in enumerate(dieCoordinates):
            if min_x in dieCoordinate[0]:
                min_x_index = dieCoordinate_index
            
            if min_y in dieCoordinate[1]:
                min_y_index = dieCoordinate_index
        
        row_2_y_start = dieCoordinates[min_y_index, 3]
        col_2_x_start = dieCoordinates[min_x_index, 2]
        
        minX = int( min_x )
        minY = int( min_y )
        maxX = int( max_x )
        maxY = int( max_y )
    
    # Changes column names in dieNames
    for box_index in range(len(dieCoordinates)):
        
        x1 = int( dieCoordinates[box_index][0] )
        y1 = int( dieCoordinates[box_index][1] )
        x2 = int( dieCoordinates[box_index][2] )
        y2 = int( dieCoordinates[box_index][3] )
        
        midX = round((x1 + x2)/2)
        midY = round((y1 + y2)/2)
        
        # Creates dieNames list row and column number
        if y1 < row_2_y_start:
            rowNumber = 1
        else:
            rowNumber = math.floor(max((y1-row_2_y_start),0)/(box_height*die_spacing_height)+2)
        # rowNumber = math.floor(max((y1-minY),0)/(box_height*die_spacing_height)+1)
        rowNumber = str(rowNumber)
        if x1 < col_2_x_start:
            colNumber = 1
        else:
            colNumber = math.floor(max((x1-col_2_x_start),0)/(box_width*die_spacing_width)+2)
        # colNumber = math.floor(max((x1-minX),0)/(box_width*die_spacing_width)+1)
        colNumber = str(colNumber)
        
        real_rowNum = (path_row_number - 1)*80 + part_row_number + int(rowNumber)
        real_colNum = (path_col_number - 1)*200 + part_col_number + int(colNumber)
        
        # Adds '0's in string if needed
        if int(real_colNum) < 10:
            real_colNum = "00" + str(real_colNum)
        elif int(real_colNum) < 100:
            real_colNum = "0" + str(real_colNum)
        
        if int(real_rowNum) < 10:
            real_rowNum = "00" + str(real_rowNum)
        elif int(real_rowNum) < 100:
            real_rowNum = "0" + str(real_rowNum)
        
        xmin = int(dieCoordinates[box_index][0])
        ymin = int(dieCoordinates[box_index][1])
        xmax = int(dieCoordinates[box_index][2])
        ymax = int(dieCoordinates[box_index][3])
        
        real_image_name = "R_{}.C_{}".format(real_rowNum, real_colNum)
        
        # For Excel sheets below
        all_dieNames.append(real_image_name)
        all_dieBinNumbers.append( int(die_class_indexes[box_index]) - 1)
        
    
    
    save_image(predicted_image/255, PREDICTED_PATH + image_name + ".jpg")
    
    # # Removes images if number of dies between a certain amount
    # if len(dieCoordinates) >= 99 and len(dieCoordinates) <= 100:
    #     os.remove(image_path)
    
    # # Removes images if doesn't have label/class wanted
    # if not (len(die_class_indexes[die_class_indexes == 6]) > 0
    #         ):
    #     os.remove(image_path)
    
    ten_scale = max(int(len(os.listdir(TO_PREDICT_PATH))*0.01), 1)
    
    ii += 1
    if ii % ten_scale == 0:
        fps_end_time = time.time()
        fps_time_lapsed = fps_end_time - fps_start_time
        fps = round(ten_scale/fps_time_lapsed, 2)
        percent_progress = round(ii/len(os.listdir(TO_PREDICT_PATH))*100)
        images_left = len(os.listdir(TO_PREDICT_PATH)) - ii
        
        time_left = images_left/(fps) # in seconds
        mins = time_left // 60
        sec = time_left % 60
        
        sys.stdout.write('\033[2K\033[1G')
        print("  " + str(percent_progress) + "%",
              "-",  fps, "FPS -",
              "Time Left: {0}m:{1}s".format(int(mins), round(sec) ),
              end="\r"
              )
        fps_start_time = time.time()
print("")
    
# XLS Section
# -----------------------------------------------------------------------------
# Deletes the "Dies" unnecessary class
del classes_1[0]

# XLSX PART
max_row = 320
max_col = 400


# Finds how many rows and columns per Excel sheet
row_per_sheet = 160
col_per_sheet = 200

# Iterates over each row_per_sheet x col_per_sheet dies 
#  and defaults bin number to 8 - Untested
for row in range(row_per_sheet):
    for col in range(col_per_sheet):
        for worksheet in worksheet_list:
            worksheet.write(row, col, 8, bin_colors_list[-1])

print("   Started writing Excel sheet bin numbers..")
# Writes all dies info in Excel
for all_dieName_index, all_dieName in enumerate(all_dieNames):
    row = int( re.findall(r'\d+', all_dieName)[0] )
    col = int( re.findall(r'\d+', all_dieName)[1] )
    
    # Checks to see which background bin number to use
    background = bin_colors_list[all_dieBinNumbers[all_dieName_index]]
    
    bin_number = all_dieBinNumbers[all_dieName_index]
    class_bin_number = bin_number
    
    # If row or col is below 10 (or 100 for SMiPE4 and similar) adds "0"s
    # SMIPE col-1 SECTION MAY NEED REEEEEEEEEEDDDDDDDDDDDDDDOOOOOOOOOOOOOOOOOOOOOOOOOOOOONNNNNNNNNNNEEEEEEE
    # --------------------------------------------------------------------
    
    # Row Section
    # THIS PART IS FOR LED 160,000 WAFER!
    row_string = str(row)
    
    # Col Section
    # THIS PART IS FOR LED 160,000 WAFER!
    col_string = str(col)
            
    # --------------------------------------------------------------------
    # Writes bin numbers in Excel sheets
    if row <= row_per_sheet:
        if col <= col_per_sheet:
            # Just writes bins
            worksheet_list[0].write(row-1, col-1, 
                                bin_number, 
                                background)
            
        else:
            # Just writes bins
            worksheet_list[1].write(row-1, col-1-col_per_sheet, 
                               bin_number,
                               background)
    else:
        if col <= col_per_sheet:
            # Just writes bins
            worksheet_list[2].write(row-1-row_per_sheet, col-1, 
                               bin_number,
                               background)
        else:
            # Just writes bins
            worksheet_list[3].write(row-1-row_per_sheet, col-1-col_per_sheet, 
                               bin_number,
                               background)


bin_count_dict = {}
for worksheet_index in range(len(worksheet_list)):
    bin_count_dict[worksheet_index] = {}
    for bin_index in range(len(classes_1)):
        bin_count_dict[worksheet_index]["bin{}".format(bin_index)] = 0

# Counts how many bins in each worksheet
for worksheet_index, worksheet in enumerate(worksheet_list):
    for row in range(row_per_sheet):
        for col in range(col_per_sheet):
            bin_num = worksheet.table[row][col].number
            if bin_num == 8:
                continue
            bin_count_dict[worksheet_index]["bin{}".format(bin_num)] += 1


# Selects appropriate "Not Tested Count" name
not_tested_name = "8 - Not_Tested-Count"

# For each sheet, writes bin class count and colors background
for worksheet_index, worksheet in enumerate(worksheet_list):
    for class_index, class_name in enumerate(classes_1):
        # Writes in bold and makes color background for each sheet a count of class bins
        worksheet.write(int(max_row/sqrt(len(worksheet_list) ) ) + 2 + class_index, 0, 
            class_name, bin_bold_colors_list[class_index]
            )
        worksheet.write(int(max_row/sqrt(len(worksheet_list) ) ) + 2 + class_index, 11, 
            bin_count_dict[worksheet_index]["bin{}".format(class_index)], 
            bin_bold_colors_list[class_index]
            )
        
        # Not Tested Count Section
        worksheet.write(int(max_row/sqrt(len(worksheet_list) ) ) + 2 + len(classes_1), 0, 
            not_tested_name, bin_bold_colors_list[-1]
            )
        worksheet.write(int(max_row/sqrt(len(worksheet_list) ) ) + 2 + len(classes_1), 11, 
            "="+str(len(all_dieNames)-1) + "/4"
            +"-sum(L{}:L{})".format((int(max_row/sqrt(len(worksheet_list) ) ) + 3),
                                    (int(max_row/sqrt(len(worksheet_list) ) ) + 2 + len(classes_1))), 
            bin_bold_colors_list[-1]
            )
        
        for index in range(10):
            worksheet.write(int(max_row/sqrt(len(worksheet_list) ) ) + 2 + class_index, (index+1), 
                "", bin_colors_list[class_index]
                )
            worksheet.write(int(max_row/sqrt(len(worksheet_list) ) ) + 2 + len(classes_1), (index+1), 
                "", bin_colors_list[-1]
                )
        
        # Writes an additional total count in case more than one sheet with total of sum of each sheet
        if len(worksheet_list) > 1:
            # Writes in bold and makes color background for each sheet a count of class bins
            worksheet.write(int(max_row/sqrt(len(worksheet_list) ) ) + 4 + len(classes_1) + class_index, 0, 
                "Total - " + class_name, 
                bin_bold_colors_list[class_index]
                )
            tot_count = 0
            for worksheet_index_v2 in range(len(worksheet_list)):
                tot_count += bin_count_dict[worksheet_index_v2]["bin{}".format(class_index)]
            worksheet.write(int(max_row/sqrt(len(worksheet_list) ) ) + 4 + len(classes_1) + class_index, 11, 
                tot_count, 
                bin_bold_colors_list[class_index]
                )
            
            # Not Tested Count Section
            worksheet.write(int(max_row/sqrt(len(worksheet_list) ) ) + 4 + len(classes_1) + len(classes_1), 0, 
                "Total - " + not_tested_name, 
                bin_bold_colors_list[-1]
                )
            worksheet.write(int(max_row/sqrt(len(worksheet_list) ) ) + 4 + len(classes_1) + len(classes_1), 11, 
                "="+str(len(all_dieNames)-1)
                +"-sum(L{}:L{})".format((int(max_row/sqrt(len(worksheet_list) ) ) + 5 + len(classes_1)),
                                        (int(max_row/sqrt(len(worksheet_list) ) ) + 4 + len(classes_1) + len(classes_1))), 
                bin_bold_colors_list[-1]
                )
            
            for index in range(10):
                worksheet.write(int(max_row/sqrt(len(worksheet_list) ) ) + 4 + len(classes_1) + class_index, (index+1), 
                    "", bin_colors_list[class_index]
                    )
                worksheet.write(int(max_row/sqrt(len(worksheet_list) ) ) + 4 + len(classes_1) + len(classes_1), (index+1), 
                    "", bin_colors_list[-1]
                    )
    
    # Sets the appropriate width for each column
    worksheet.set_column(0, (col_per_sheet), width=round((20*max_row/max_col)*.12, 2) )
    
    if len(worksheet_list) > 1: 
        worksheet.set_column(11, 11, width=8)
    elif len(all_dieNames) < 1000:
        worksheet.set_column(11, 11, width=3.5)
    else:
        worksheet.set_column(11, 11, width=7)
    
    # Sets zoom
    worksheet.set_zoom( max( int(2080.6*(max_row/sqrt(len(worksheet_list)))**-0.867 ) , 20 ) )
                
                
            
workbook.close()
        # -----------------------------------------------------------------------------
    
    
    
    

print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)

# =============================================================================