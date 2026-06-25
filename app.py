import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from search import search
import os
import json
import faiss
import torch
from transformers import AutoImageProcessor, AutoModel

# Cache the YOLO model so it is only loaded once
@st.cache_resource
def load_yolo_model():
    return YOLO("model/best.pt")

@st.cache_resource
def load_catalog_file_names(search_engine_folder):
    with open(os.path.join(search_engine_folder,"index.json")) as f:
        catalog_file_names=json.load(f)
    return catalog_file_names

@st.cache_resource
def load_index(search_engine_folder):
    return faiss.read_index(os.path.join(search_engine_folder,"vector.index")) 

@st.cache_resource
def load_processor(model_variant):
    return AutoImageProcessor.from_pretrained(f"facebook/{model_variant}")

@st.cache_resource
def load_search_model(device,model_variant):
    return AutoModel.from_pretrained(f"facebook/{model_variant}").to(device)

# Dummy function to simulate image search
# def search_images(uploaded_image):
#     # In a real application, implement your image search logic here
#     # For demonstration, we return dummy images and labels

    

#     result_images = [
#         "data/panels/train/DS1BIMG67_png.rf.8f5af3c4204d4620c37916a99d2649f4.png",
#         "data/panels/train/DS1BIMG67_png.rf.8f5af3c4204d4620c37916a99d2649f4.png",
#         "data/panels/train/DS1BIMG67_png.rf.8f5af3c4204d4620c37916a99d2649f4.png"
#     ]
#     labels = ["Label 1", "Label 2", "Label 3"]
#     return result_images, labels

def main():
    st.title("Altnova Balcony Search Application")

    SEARCH_ENGINE_FOLDER="search_engine"
    NR_RESULTS = 3

    #load catalog image names
    catalog_file_names=load_catalog_file_names(SEARCH_ENGINE_FOLDER)
    index= load_index(SEARCH_ENGINE_FOLDER)
    
    MODEL_VARIANT="dinov2-base"

    #load the model and processor
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    processor = load_processor(MODEL_VARIANT)
    model = load_search_model(device,MODEL_VARIANT)

    panel_model = load_yolo_model()
    
    uploaded_file = st.file_uploader("Upload an Image (png or jpg)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        #result_images, labels = search_images(image)
        result_images, labels = search(panel_model,device,processor,model,image,index,catalog_file_names)
        
        st.subheader("Search Results:")
        cols = st.columns(NR_RESULTS)
        for i in range(len(result_images)):
            with cols[i]:
                #st.image(result_images[i], caption=labels[i], width=300)
                st.image(result_images[i], width=300)
                st.markdown(f"<h3 style='text-align: left;'>balcony type: {labels[i]}</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()