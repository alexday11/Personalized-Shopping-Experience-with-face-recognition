import cv2
import pickle
import streamlit as st
from Function import recommend_retail , open_camera
from columns import create_column_gen


st.set_page_config(
    page_title="Personalized Shopping Experience",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title('Personalized Shopping Experience')

#st.image('./Image/Background.jpg')
# Columns Button 
st.video('./Image/background.mp4')

### Image #########


st.header('มาดูกันว่าวันนี้เราสามารถจะแนะนำอะไรให้คุณได้บ้าง ??',divider='red')

st.subheader('Personalized product recommendations with Camera')
#st.image('./Image/women.jpg',use_column_width=True)
if st.button('Recommend Products with Camera'):
   results,names = open_camera()
   st.subheader('Hi !! {} มาดูสินค้าที่เราแนะนำให้คุณวันนี้กัน.....'.format(names))
   create_column_gen(results)


st.subheader('Personalized product recommendations with Image')
#st.image('./Image/selfie.jpg',use_column_width=True)
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    result,name_img = recommend_retail(uploaded_image)
    st.subheader('Hi !! {} มาดูสินค้าที่เราแนะนำให้คุณวันนี้กัน.....'.format(name_img))
    create_column_gen(result)
