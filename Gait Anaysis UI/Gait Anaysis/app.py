import streamlit as st
import pickle
from key_array import key_array
from dynamic_for_loop_front import dynamic_front
from dynamic_for_loop_side import dynamic_side
from static_front import static_front
from static_side import static_side
from feature_gen import feature_gen
from predict import predict_
import base64
import subprocess
import time 

st.header("Person Identification System using Gait Pattern Analysis")

st.markdown('---')
cam_angle = st.sidebar.selectbox("Select Camera Angle", ["None", "Front View", "Side View"])
if cam_angle == "Front View":
    cam_angle = 'front'
if cam_angle == "Side View":
    cam_angle = 'side'

uploaded_video = st.sidebar.file_uploader("Upload Suspect Video", type=["mp4"])
if uploaded_video is not None:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_video.getbuffer())

    key_array('uploaded_video.mp4', cam_angle)

    if cam_angle == "front":
        f_file = open('valid_key_arrays/key_array_evidance_front.pkl', 'rb')
        key_array = pickle.load(f_file)
        dynamic_front(key_array)
        static_front(key_array)

        feature_gen(cam_angle)
        label = predict_(cam_angle)
        print(label)

    elif cam_angle == "side":
        f_file = open('valid_key_arrays/key_array_evidance_side.pkl', 'rb')
        key_array = pickle.load(f_file)
        dynamic_side(key_array)
        static_side(key_array)

        feature_gen(cam_angle)
        label = predict_(cam_angle)
        print(label)

    temp_file_result = "./output.mp4"
    convertedVideo = "./testh264.mp4"
    subprocess.call(args=f"ffmpeg -y -i {temp_file_result} -c:v libx264 {convertedVideo}".split(" "))

    with open('testh264.mp4', "rb") as f:
        video_content = f.read()

    video_str = f"data:video/mp4;base64,{base64.b64encode(video_content).decode()}"
    st.markdown(f"""
        <video style="display: block; margin: auto; width: 700px;" control autoplay>
            <source src="{video_str}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)

    st.markdown('---')

    col1, col2, col3 = st.columns(3)

    with col1:
        pass
    with col2:
        time.sleep(10)
        st.subheader("Identified Suspect")
        st.image('suspects/leble_{}.png'.format(label))
        st.text("Predicted Label = {}".format(label))
        #st.text("Predicted Label = {}".format(label), style="font-size: 18px;")


    with col3:
        pass
