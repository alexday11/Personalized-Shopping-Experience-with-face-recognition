import streamlit as st
import pandas as pd
import pickle

def create_column_gen(names1):
   num_rows = 5
   num_cols = 5

   for row in range(num_rows):
      cols = st.columns(num_cols)
      for col_index, col in enumerate(cols):
         movie_index = row * num_cols + col_index
         if movie_index < len(names1):
            # You can add images to each column using st.image
            image_url = "https://static.streamlit.io/examples/owl.jpg"  # Replace with your image URL
                
            # Use HTML to overlay text on the image
            col.image(image_url, use_column_width=True)
            col.markdown(f"<p style='text-align: center; font-size: 18px;'>{names1[movie_index]}</p>", unsafe_allow_html=True)
