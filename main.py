from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

feature_list=np.array(pickle.load(open('featurevector.pkl','rb')))
filenames=np.array(pickle.load(open('filenames.pkl','rb')))

model=ResNet50(weights='imagenet',input_shape=(224,224,3))
model_new=Model(model.input,model.layers[-2].output)

def preprocess_img(img,model):
    img=cv2.imread(img)
    img=cv2.resize(img,(224,224))
    img=np.array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    img=model.predict(img).flatten()
    img=img/norm(img)

    return img

img=preprocess_img(r'dataset\1636.jpg',model_new)

neighbors=NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

distance,indices=neighbors.kneighbors([img])

for file in indices[0][1:5]:
    imgName=cv2.imread(filenames[file])
    cv2.imshow('Frame',cv2.resize(imgName,(640,480)))
    cv2.waitKey(0)