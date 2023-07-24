import pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math
import tensorflow as tf
from tensorflow import keras
from csv import writer
from collections import deque
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os
import csv
import pandas as pd
from tensorflow.python.keras.saving.hdf5_format import save_attributes_to_hdf5_group

def predict_(cam_angle):



    filename= open('model/model_results_{}.pkl'.format(cam_angle), 'rb')
    print(filename)
    model = pickle.load(filename)

    file_path = 'valid_feature_vector/FV_evidance_{}.pkl'.format(cam_angle)

    temp_3 = open(file_path, 'rb')
    df_pre = pickle.load(temp_3)


    temp_3.close()
    #print(name, vid_no, condition, d, "Actual ", idx)

    #df_pre = df_pre.drop(df_pre.iloc[:, -1:], axis=1)
    df_eval = df_pre
    X = df_eval

    #a = model.evaluate(X, y)
    l = model.predict(X)

    df_l = pd.DataFrame(l)
    column_averages = df_l.mean()
    # print(df_l)

    predicted_label = np.argmax(column_averages)
    
    return predicted_label

