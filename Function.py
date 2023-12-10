import os
import cv2
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from keras_facenet import FaceNet
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder ,Normalizer


# Load Data
facenet_model = FaceNet()
detector = cv2.CascadeClassifier('./Dataset/haar.xml')
model_face = pickle.load(open('./Dataset/model.p','rb'))
model_recommend = Word2Vec.load('./Dataset/amazon_store.model')
data = np.load('./Face_npz/FaceEmbedHarr.npz')
database = pickle.load(open('./Dataset/Database.pkl','rb'))
dataset_amazon = pd.read_csv('./Dataset/dataset_amazom.csv')
dataset_amazon['Customer ID'] = dataset_amazon['Customer ID'].astype(str)
products_dict = pickle.load(open('product_dict.pkl','rb'))

# Preprocess Data

in_coder = Normalizer()
out_coder = LabelEncoder()

emd_trainX,trainy,emd_testX,testy = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

# Normalizer
emd_trainX_norm = in_coder.fit_transform(emd_trainX)
emd_testX_norm = in_coder.transform(emd_testX)

# FIT
out_coder.fit(trainy)
trainy_enc = out_coder.transform(trainy)
testy_enc = out_coder.transform(testy)

### Image #########

def extract_face4(file,resize=(160,160)):
    try:
        img = Image.open(file)
        img = np.array(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f'Error processing uploaded image: {str(e)}')
        return None
    
    faces = detector.detectMultiScale(gray_img, 1.1, 5)
    if len(faces) == 0:
        return None
    
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    img_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    img_face = img_face.resize(resize)
    face_arr = np.asarray(img_face)
    return face_arr


def get_embedding(facenet_model , face):
    face_pixel = face.astype('float32')
    samples =  np.expand_dims(face_pixel,axis=0)
    embed = facenet_model.embeddings(samples)
    return embed[0]

def prediction(face):
    img = extract_face4(face)
    embed = get_embedding(facenet_model,img)
    embed = np.expand_dims(embed,axis=0)
    embed_norm = in_coder.transform(embed)
    index = model_face.predict(embed_norm)
    predict_names = out_coder.inverse_transform(index)
    return predict_names[0]

def recommend_retail(filename):
    product_rec = []
    try:
        predict_names = prediction(filename)
        check_id = database[database['Names'] == predict_names]['Customer ID'].values[0]
        sku = dataset_amazon[dataset_amazon['Customer ID'] == check_id]['StockCode'].values[:5]
        for i in range(len(sku)):
            try:
                similars = model_recommend.wv.most_similar(sku[i], topn=5)
                for j in similars:
                    if j[1] > 0.5:
                        product_rec.append(products_dict[j[0]][0])
            except KeyError:
                # Handle the KeyError here, e.g., you can print a message or simply continue the loop.
                print(f"KeyError for SKU: {sku[i]}. Skipping...")
                st.write('Sorry not found SKU')
                continue
    except AttributeError as e:
        print('Sorry Can not detect face')
        st.write('Sorry Can not detect face')
        return product_rec
    return product_rec , predict_names


# Video

def extract_face2(img,re_size=(160,160)):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detectMultiScale(gray_img,1.1,5)
    if len(faces) == 0:
        return None
    x,y,w,h = faces[0]
    face = img[y:y+h,x:x+w]
    img_face = Image.fromarray(cv2.cvtColor(face,cv2.COLOR_BGR2RGB))
    img = img_face.resize(re_size)
    face_arr = np.asarray(img)
    return face_arr

def get_embedding2(facenet_model, face):
    if face is None:
        return None
    
    face_pixel = face.astype('float32')
    samples = np.expand_dims(face_pixel, axis=0)
    embedd = facenet_model.embeddings(samples)
    return embedd[0]




def open_camera():
    cap = cv2.VideoCapture(0)
    stop_button_pressed = st.button('stop')
    frame_placeholder = st.empty()
    names_product = []
    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()
        if ret == True:
            face = extract_face2(frame)
            face_embed = get_embedding2(facenet_model, face)
            #cv2.imshow('output', frame)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame,channels='RGB')
            if face_embed is not None:
                samples = np.expand_dims(face_embed,axis=0)
                index = model_face.predict(samples)
                proba = model_face.predict_proba(samples)
                class_prob = proba[0,index[0]]
                if class_prob >= 0.5:
                    predict_names = out_coder.inverse_transform(index)
                    check_id = database[database['Names']== predict_names[0]]['Customer ID'].values[0]
                    sku = dataset_amazon[dataset_amazon['Customer ID']==check_id]['StockCode'].values[:5]
                    #print('Hi... {}'.format(predict_names[0]))
                    for i in range(len(sku)):
                        try:
                            similars = model_recommend.wv.most_similar(sku[i],topn=5)
                            for j in similars:
                                if j[1] > 0.6:
                                #print('{:6} {:36} {:.3f}'.format(j[0],products_dict[j[0]][0],j[1]))
                                    names_product.append(products_dict[j[0]][0])
                        except KeyError:
                            # Handle the KeyError here, e.g., you can print a message or simply continue the loop.
                            print(f"KeyError for SKU: {sku[i]}. Skipping...")
                            continue    
                    break
            
            if cv2.waitKey(1) & 0xFF == ord('d') or stop_button_pressed:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return names_product , predict_names[0]