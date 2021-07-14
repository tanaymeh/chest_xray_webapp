import streamlit as st
import PIL
import numpy as np
import torch
import time
import os
import matplotlib.image as mpimg
from fastai.vision import open_image, load_learner, image, torch
from PIL import Image, ImageOps
from predict import infer

st.title("Chest X-Ray Classification Application")
st.header("Classification Example")

option = st.radio('', ['Choose a Sample XRay', 'Upload your own XRay'])

if option == 'Choose a Sample XRay':
    # Get a list of test images in the folder
    test_imgs = os.listdir("test_imgs/")
    test_img = st.selectbox(
        'Please Select a Test Image:',
        test_imgs
    )
    
    # Display and then predict on that image
    fl_path = "test_imgs/"+test_img
    img = open_image(fl_path)
    
    display_img = mpimg.imread(fl_path)
    st.image(display_img, caption="Chosen XRay", use_column_width=True)
    
    st.write("")
    with st.spinner("Identifying the Disease..."):
        time.sleep(5)
    label, prob = infer(img)
    st.success(f"Image Disease: {label}, Confidence: {prob:.2f}%")

elif option == 'Upload your own XRay':
    uploaded_file = st.file_uploader("Choose an Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded XRay", use_column_width=True)
        
        img = image.pil2tensor(img, np.float32).div_(255)
        img = image.Image(img)
        st.write("")
        with st.spinner("Identifying the Disease..."):
            time.sleep(5)
        label, prob = infer(img)
        
        st.success(f"Image Disease: {label}, Confidence: {prob:.2f}%")

st.warning("NOTE: If you upload an Image which is not a Chest XRay, the model will give very wierd predictions because it's trained to identify which one of the 2 labels the model is most confident of.")


st.write("Project by Team: FeedForwardNN")
