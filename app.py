import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
import numpy as np
import os
import pickle
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from PIL import Image

feature_list=np.array(pickle.load(open('featurevector.pkl','rb')))
filenames=np.array(pickle.load(open('filenames.pkl','rb')))

model=ResNet50(weights='imagenet',input_shape=(224,224,3))
model_new=Model(model.input,model.layers[-2].output)

st.title('Fashion Image Search Engine')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def preprocess_img(img,model):
    img=cv2.imread(img)
    img=cv2.resize(img,(224,224))
    img=np.array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    img=model.predict(img).flatten()
    img=img/norm(img)

    return img

def recommend(features,feature_list):
    neighbors=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)

    distance,indices=neighbors.kneighbors([features])

    return indices

uploaded_file=st.file_uploader('Choose an image')
print(uploaded_file)
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image=Image.open(uploaded_file)
        resized_image=display_image.resize((200,200))
        st.image(resized_image)

        features=preprocess_img(os.path.join('uploads',uploaded_file.name),model_new)

        indices=recommend(features,feature_list)
        
        cols = st.columns(5)
        col1 = cols[0]
        col2 = cols[1]
        col3 = cols[2]
        col4 = cols[3]
        col5 = cols[4]

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header('Some error occured in file upload')
