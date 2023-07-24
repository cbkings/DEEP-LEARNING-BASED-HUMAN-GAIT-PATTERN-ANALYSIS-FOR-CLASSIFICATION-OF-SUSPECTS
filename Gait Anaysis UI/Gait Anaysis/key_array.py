import mediapipe as mp
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import itertools as itr
import pprint, pickle
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


# Create a Tkinter root window

# Open a file dialog box to select the video file
# video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])

# #video_path="/home/cbk/Music/Prediction_video_set/fps_30/front/normal/yohan_normal_front_3.mp4"

def key_array(video_path, cam_angle):

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', codec, fps, (width, height))
    # Get total number of frames in the video

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid = cap.get(cv2.CAP_PROP_FPS)

    myarray = np.empty(shape=(0,33,5))  # initialize with empty array

    # Initiate pose model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        idx=0
        while cap.isOpened():

            ret, frame = cap.read()
            if ret:

                # Get the current frame number
                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                #print("Frame number:", frame_num)

                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Make Detections
                results = pose.process(image)

                # Recolor image back to BGR for rendering
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if results.pose_landmarks != None:
                    key_list= []
                    t=frame_num/fps_vid
                    #print(t)
                    for data_point in results.pose_landmarks.landmark:
                        key_list.append([data_point.x,data_point.y,data_point.z,t,data_point.visibility])    
                    mat = np.array(key_list)
                    myarray = np.append(myarray,[mat],axis=0)  # append mat to myarray without dummy row

                # Display the resulting frame
                #cv2.imshow('Raw Webcam Feed', image)
                out.write(image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

    key_array = myarray  # assign myarray to key_array variable


    # assume you have a variable called `myarray` that you want to save as a .pkl file
    output_path = open('valid_key_arrays/key_array_evidance_{}.pkl'.format(cam_angle), 'wb')


    pickle.dump(key_array, output_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    #out.release()
    #cv2.destroyAllWindows()