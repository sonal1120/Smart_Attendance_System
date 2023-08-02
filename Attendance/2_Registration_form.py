import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(page_title='Registration Form')
st.subheader('Registration Form')

# Initialization of registration form
registration_form = face_rec.RegistrationForm()


# Step 1: Collect person name and role
# Form
person_name = st.text_input(label='Name', placeholder='First & Last Name')
role = st.selectbox(label='Select your Role', options=('Student', 'Teacher'))


# Step 2: Collect Facial embeddings(Face Sample) of person
def video_frame_callback(frame):
    img = frame.to_ndarray(format='bgr24')  # 3-dimensional array (BGR)
    reg_img, embedding = registration_form.get_embeddings(img)
    
    # Two-step process
    # 1st step is to save data into the local computer txt
    if embedding is not None:
        with open('face_embedding.txt', mode='ab') as f:  # ab = append the values in bytes
            np.savetxt(f, embedding)

    return av.VideoFrame.from_ndarray(reg_img, format='bgr24')


webrtc_streamer(key='registration', video_frame_callback=video_frame_callback)


# Step 3: Save the data in Redis database
if st.button('Submit'):
    return_val = registration_form.save_data_in_redis_db(person_name, role)
    if return_val is True:
        st.success(f"{person_name} registered successfully")
    elif return_val == 'name_false':
        st.error('Please enter the name: Name cannot be empty or contain only spaces')
    elif return_val == 'file_false':
        st.error('face_embedding.txt is not found. Please refresh the page and execute')
