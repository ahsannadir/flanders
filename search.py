#***********************
# library for search
#***********************


import faiss
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
#from PIL import Image
import os
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import ImageGrid
#import seaborn_image as si
#from seaborn_image import ImageGrid
import cv2
import json


#from ultralytics import YOLO

from pathlib import Path
import re

def crop_panel(model,input_img):
    result= model.predict(input_img)[0] 
    img = np.copy(result.orig_img)
    if result.masks is not None and len(result.masks.xy) > 0: #there is a detection
        #keep only first mask/box if multiple detected
        img_name = Path(result.path).stem
        print(img_name)
        b_mask = np.zeros(img.shape[:2], np.uint8)


        # Create contour mask 
        contour = result.masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (1, 1, 1), cv2.FILLED)


        # plt.figure()
        # plt.axis("off")
        # plt.imshow(b_mask * 255)

       

        # Isolate object with black background
        # mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        # isolated = cv2.bitwise_and(mask3ch, img)

        masked_img = cv2.bitwise_and(img,img,mask=b_mask)
        # plt.figure()
        # plt.axis("off")
        # plt.imshow(masked_img)

        # detection crop
        x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[0].astype(np.int32)
        cropped_input = masked_img[y1:y2, x1:x2]

        cropped_input = cv2.resize(cropped_input, (360, 900)) #make sure the cropped input has the same size as the catalog images
        # plt.figure()
        # plt.axis("off")
        # plt.imshow(cropped_input)

        #segment_panel(cropped_input)

        #segment_panel_sam2(cropped_input)
        #for debugging here
        # Save isolated object to file
        #_ = cv2.imwrite(f"data/output/{img_name}.png", cropped_input)

        _ = cv2.imwrite("data/output/crop.png", cropped_input)

    else:
        #give the original image as the panel
        img = cv2.resize(img, (360, 900))
        cv2.imwrite("data/output/crop.png", img)    


def get_type(names,nr_results):
    types=[]
    pattern=re.compile(r'DS(\w+)IMG.*')
    for i in range(nr_results):
        name=Path(names[i]).stem
        #print(name)
        m=pattern.match(name)
        if m is not None:
            types.append(m.group(1))
    return types


def search(panel_model,device,processor,model,image,index,catalog_file_names,NR_SEARCH_RESULTS=3):
    """image is a PIL image"""
    #input image
    #image = Image.open(os.path.join(DATA_FOLDER,subdir,image_name))
    #image=cv2.cvtColor(cv2.imread(image_name),cv2.COLOR_BGR2RGB)
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(image)


    crop_panel(panel_model,image)

    image=cv2.cvtColor(cv2.imread("data/output/crop.png"),cv2.COLOR_BGR2RGB)
    
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(image)


    #Extract the features
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

    #Normalize the features before search
    embeddings = outputs.last_hidden_state
    embeddings = embeddings.mean(dim=1)
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)

    #perform search of top N images
    dist,idx = index.search(vector,NR_SEARCH_RESULTS)
    #print('distances:', dist, 'indexes:', idx)


    #load catalog images
    #images= [Image.open(catalog_file_names[i]).convert('RGB') for i in idx[0]]
    
    #images= [cv2.cvtColor(cv2.imread(catalog_file_names[i]),cv2.COLOR_BGR2RGB) for i in idx[0]]
    names=[catalog_file_names[i] for i in idx[0]]

   
    types=get_type(names,NR_SEARCH_RESULTS)
    return(names,types)