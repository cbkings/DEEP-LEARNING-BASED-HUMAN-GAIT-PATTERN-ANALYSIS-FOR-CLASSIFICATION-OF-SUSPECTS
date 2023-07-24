import mediapipe as mp
import cv2
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.misc import derivative
from scipy.signal import find_peaks
from statistics import stdev,mean
import itertools as itr
import math
import pprint, pickle

def static_front(key_array):
    def length(x1,x2,y1,y2,z1,z2):
        output=(math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2))
        return output
    def thigh_r(k):
        l_xy=length(key_array[k][24][0],key_array[k][26][0],key_array[k][24][1],key_array[k][26][1],0,0)
        visi=key_array[k][24][4]*key_array[k][26][4]
        return l_xy,visi

    def thigh_l(k):
        l_xy=length(key_array[k][23][0],key_array[k][25][0],key_array[k][23][1],key_array[k][25][1],0,0)
        visi=key_array[k][23][4]*key_array[k][25][4]
        return l_xy,visi

    def shank_r(k):
        l_xy=length(key_array[k][28][0],key_array[k][26][0],key_array[k][26][1],key_array[k][28][1],0,0)
        visi=key_array[k][28][4]*key_array[k][26][4]
        return l_xy,visi

    def shank_l(k):
        l_xy=length(key_array[k][27][0],key_array[k][25][0],key_array[k][27][1],key_array[k][25][1],0,0)
        visi=key_array[k][27][4]*key_array[k][25][4]
        return l_xy,visi

    def ankle_to_heel_r(k):
        l_xy=length(key_array[k][28][0],key_array[k][30][0],key_array[k][28][1],key_array[k][30][1],0,0)
        visi=key_array[k][28][4]*key_array[k][30][4]
        return l_xy,visi

    def ankle_to_heel_l(k):
        l_xy=length(key_array[k][27][0],key_array[k][29][0],key_array[k][27][1],key_array[k][29][1],0,0)
        visi=key_array[k][27][4]*key_array[k][29][4]
        return l_xy,visi

    def ubl(k):
        l_xy=upper_body(key_array[k][11][0],key_array[k][12][0],key_array[k][23][0],key_array[k][24][0],key_array[k][11][1],key_array[k][12][1],key_array[k][23][1],key_array[k][24][1],0,0,0,0)
        return l_xy
    def upper_body(x11,x12,x23,x24,y11,y12,y23,y24,z11,z12,z23,z24):
        output=math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2+(((z11+z12)/2)-((z23+z24)/2))**2)
        return output
    def center(k):
        cent=(key_array[k][11][0]+key_array[k][12][0]+key_array[k][23][0]+key_array[k][24][0])/4
        return cent

    def shoulder(k):
        l_xy=length(key_array[k][11][0],key_array[k][12][0],key_array[k][11][1],key_array[k][12][1],0,0)
        visi=key_array[k][11][4]*key_array[k][12][4]
        return l_xy,visi

    def hip_size(k):
        l_xy=length(key_array[k][24][0],key_array[k][23][0],key_array[k][24][1],key_array[k][23][1],0,0)
        visi=key_array[k][24][4]*key_array[k][23][4]
        return l_xy,visi

    def upper_arm_r(k):
        l_xy=length(key_array[k][12][0],key_array[k][14][0],key_array[k][12][1],key_array[k][14][1],0,0)
        visi=key_array[k][12][4]*key_array[k][14][4]
        return l_xy,visi

    def upper_arm_l(k):
        l_xy=length(key_array[k][11][0],key_array[k][13][0],key_array[k][11][1],key_array[k][13][1],0,0)
        visi=key_array[k][11][4]*key_array[k][13][4]
        return l_xy,visi

    def lower_arm_r(k):
        l_xy=length(key_array[k][16][0],key_array[k][14][0],key_array[k][16][1],key_array[k][14][1],0,0)
        visi=key_array[k][16][4]*key_array[k][14][4]
        return l_xy,visi

    def lower_arm_l(k):
        l_xy=length(key_array[k][15][0],key_array[k][13][0],key_array[k][15][1],key_array[k][13][1],0,0)
        visi=key_array[k][15][4]*key_array[k][13][4]
        return l_xy,visi

    def palm_to_finger_r(k):
        l_xy=length(key_array[k][16][0],key_array[k][20][0],key_array[k][16][1],key_array[k][20][1],0,0)
        visi=key_array[k][16][4]*key_array[k][20][4]
        return l_xy,visi

    def palm_to_finger_l(k):
        l_xy=length(key_array[k][15][0],key_array[k][19][0],key_array[k][15][1],key_array[k][19][1],0,0)
        visi=key_array[k][15][4]*key_array[k][19][4]
        return l_xy,visi

    def face_width(k):
        l_xy=length(key_array[k][7][0],key_array[k][8][0],key_array[k][7][1],key_array[k][8][1],0,0)
        visi=key_array[k][7][4]*key_array[k][8][4]
        return l_xy,visi

    def mouth_width(k):
        l_xy=length(key_array[k][9][0],key_array[k][10][0],key_array[k][9][1],key_array[k][10][1],0,0)
        visi=key_array[k][9][4]*key_array[k][10][4]
        return l_xy,visi

    def eye_size(k):
        l_xy=length(key_array[k][4][0],key_array[k][6][0],key_array[k][4][1],key_array[k][6][1],0,0)
        visi=key_array[k][4][4]*key_array[k][6][4]
        return l_xy,visi

    def eyes_midpoints(k):
        l_xy=length(key_array[k][5][0],key_array[k][2][0],key_array[k][5][1],key_array[k][2][1],0,0)
        visi=key_array[k][5][4]*key_array[k][2][4]
        return l_xy,visi

    def get_peaks(df,y,p):
        peak, _ = find_peaks(y,prominence=p)
        #plt.figure(figsize=(20,5))
        #plt.xlim(0, 50)
        #plt.ylim(0, 60)
        #plt.plot(df.t,y)
        #plt.plot(df.t[peak], y[peak], "X",markerfacecolor='red',markersize=8)
        #plt.show()
        return peak

    def remove_distinct_peaks(array,df,Time_to_index_ratio,y,p,m):
        modified_mean= mean(array)
        Distance=Time_to_index_ratio*modified_mean*m
        
        peaks, _ = find_peaks(y,prominence=p,distance=Distance)
        #plt.figure(figsize=(20,5))
        #plt.xlim(0, 50)
        #plt.plot(df.t,y)
        #plt.plot(df.t[peaks], y[peaks],"X",markerfacecolor='red',markersize=8)
        #plt.show()
        return peaks

    def time_diff(peaks, df):
        array=[]
        for i in range(len(peaks)-1):
            td=df.t[peaks[i+1]]-df.t[peaks[i]]
            array.append(td)
        return array

    def step_width(x11,x12,x23,x24,y11,y12,y23,y24,x29,x30):
        step_width=abs(x30 - x29)
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output = step_width/body_length
        #output = step_wid
        return output

    def pelvic_obliquity(x23,x24,y23,y24):
        output=math.atan((abs(y23-y24))/(abs(x23-x24)))
        return output

    def foot_lifting(y28,y27,x11,x12,x23,x24,y11,y12,y23,y24):
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output=(y28 - y27)/body_length
        return output

    def step_length(x11,x12,x23,x24,y11,y12,y23,y24,z27,z28):
        step_length=abs(z28 - z27)
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output = step_length/body_length
        #output = step_len
        return output

    ##per step 
    def calculate_averages(indices, lbl_r):
        result=[]
        time=[]
        for i in range(0, len(indices)-1):
            start_idx = indices[i]
            end_idx = indices[i+1]
            t1=df_para.t[start_idx]
            #print('t1',t1)
            avg = lbl_r.iloc[start_idx:end_idx+1].mean()
            #print(avg)
            result.append(avg)
            time.append(t1)

        #print('results',result)        
        out=[result,time]
        df_out=pd.DataFrame(out)
        df_out=df_out.transpose()
        return df_out

    ##per step 
    def calculate_averages_visi(indices, lbl_r):
        result=[]
        for i in range(0, len(indices)-1):
            start_idx = indices[i]
            end_idx = indices[i+1]
            t1=df_para.t[start_idx]
            #print('t1',t1)
            avg = lbl_r.iloc[start_idx:end_idx+1].mean()
            #print(avg)
            result.append(avg)
        #print('results',result, t1) 
        out=[result]
        df_out=pd.DataFrame(out)
        df_out=df_out.transpose()
        return result

#f_file = open('valid_key_arrays/key_array_evidance_front.pkl', 'rb')
#print(f_file)



    ##Upper body height
    H=[]
    for i in range(len(key_array)):
        #time
        t=key_array[i][12][3]

        #upper body height
        x1,x2 = (((key_array[i][11][0]+key_array[i][12][0])/2),((key_array[i][23][0]+key_array[i][24][0])/2))
        y1,y2 = (((key_array[i][11][1]+key_array[i][12][1])/2),((key_array[i][23][1]+key_array[i][24][1])/2))
        #z1,z2 = (((key_array[i][11][2]+key_array[i][12][2])/2),((key_array[i][23][2]+key_array[i][24][2])/2))
        ubh = math.sqrt(((x1-x2)**2)+((y1-y2)**2))
        #ubh = math.sqrt(((x1-x2)**2)+((y1-y2)**2)+((z1-z2)**2))

        #shoulder length
        x_sl1,x_sl2 = (key_array[i][11][0],key_array[i][12][0])
        y_sl1,y_sl2 = (key_array[i][11][1],key_array[i][12][1])
        #z_sl1,z_sl2 = (key_array[i][11][2],key_array[i][12][2])
        sl=math.sqrt(((x_sl1-x_sl2)**2)+((y_sl1-y_sl2)**2))
        #sl=math.sqrt(((x_sl1-x_sl2)**2)+((y_sl1-y_sl2)**2)+((z_sl1-z_sl2)**2))

        #center of upper body in depth
        center=(key_array[i][11][2]+key_array[i][12][2]+key_array[i][23][2]+key_array[i][24][2])/4

        #right and left shoulders
        right=key_array[i][12][0]
        left=key_array[i][11][0]

        H.append([ubh])
        H[i].append(t)
        H[i].append(center)
        H[i].append(right)
        H[i].append(left)
        H[i].append(sl)
        H[i].append(sl/ubh)


    H_array = np.array(H)
    #print(H_array)

    df = pd.DataFrame(H_array,columns=['ubh','t','center','r_sdr','l_sdr','sl','sln'])
    df_ubh = pd.DataFrame(H_array,columns=['ubh','t','sl','sln','center','r_sdr','l_sdr'])

    df_upper_body_height=df_ubh.iloc[:,0]

    from scipy.interpolate import UnivariateSpline

    spl = UnivariateSpline(df.t, df.r_sdr, k=4, s=0)
    r_sdr_diff= spl.derivative()
    spl = UnivariateSpline(df.t, df.l_sdr, k=4, s=0)
    l_sdr_diff= spl.derivative()

    ##Length of lower body
    lb=[]
    for i in range(len(key_array)):
        t=key_array[i][11][3]
        l_xy=(thigh_r(i)[0]+shank_r(i)[0])/ubl(i)
        l_xy_visi=(thigh_r(i)[1]*shank_r(i)[1])
        #l_xy=thigh_r(i)+shank_r(i)
        right = key_array[i][12][0]
        left = key_array[i][11][0]
        lb.append([l_xy])
        lb[i].append(right)
        lb[i].append(left)
        lb[i].append(t)
        lb[i].append(l_xy_visi)
    ratio_ub_lb=np.array(lb) 
    df_lb = pd.DataFrame(ratio_ub_lb,columns=['l_xy','r_sdr','l_sdr','t','visi_l_xy'])

    df_lower_body_length=df_lb.iloc[:,0]
    df_lower_body_length_visi=df_lb.iloc[:,-1]

    ##Shoulder length
    sl=[]
    for i in range(len(key_array)):
        t=key_array[i][11][3]
        #l_xy=shoulder(i)
        l_xy=shoulder(i)[0]/ubl(i)
        l_xy_visi=shoulder(i)[1]
        right = key_array[i][12][0]
        left = key_array[i][11][0]
        sl.append([l_xy])
        sl[i].append(right)
        sl[i].append(left)
        sl[i].append(t)
        sl[i].append(l_xy_visi)
    shoulder_length=np.array(sl) 
    df_shoulder_l = pd.DataFrame(shoulder_length,columns=['l_xy','r_sdr','l_sdr','t','l_xy_visi'])

    df_shoulder_length=df_shoulder_l.iloc[:,0]
    df_shoulder_length_visi=df_shoulder_l.iloc[:,-1]

    ##Hip size
    hip=[]
    for i in range(len(key_array)):
        t=key_array[i][23][3]
        def hip_size(k):
            l_xy=length(key_array[k][24][0],key_array[k][23][0],key_array[k][24][1],key_array[k][23][1],0,0)
            visi=key_array[k][24][4]*key_array[k][23][4]
            return l_xy,visi

        l_xy=hip_size(i)[0]/ubl(i)
        l_xy_visi=hip_size(i)[1]

        #l_xy=hip_size(i)
        right = key_array[i][12][0]
        left = key_array[i][11][0]
        hip.append([l_xy])
        hip[i].append(right)
        hip[i].append(left)
        hip[i].append(t)
        hip[i].append(l_xy_visi)
    hip_size=np.array(hip) 
    df_hip = pd.DataFrame(hip_size,columns=['l_xy','r_sdr','l_sdr','t','l_xy_visi'])

    df_hip_size=df_hip.iloc[:,0]
    df_hip_size_visi=df_hip.iloc[:,-1]

    ## Length of upper arm
    u_arm=[]
    for i in range(len(key_array)):
        t=key_array[i][12][3]
        l_xy_r=upper_arm_r(i)[0]/ubl(i)
        l_xy_l=upper_arm_l(i)[0]/ubl(i)
        l_xy_r_visi=upper_arm_r(i)[1]
        l_xy_l_visi=upper_arm_l(i)[1]

        #l_xy_r=upper_arm_r(i)
        #l_xy_l=upper_arm_l(i)
        right = key_array[i][12][0]
        left = key_array[i][11][0]
        u_arm.append([l_xy_r])
        u_arm[i].append(l_xy_l)
        u_arm[i].append(right)
        u_arm[i].append(left)
        u_arm[i].append(t)
        u_arm[i].append(l_xy_r_visi)
        u_arm[i].append(l_xy_l_visi)
    upper_arm=np.array(u_arm) 
    df_u_arm = pd.DataFrame(upper_arm,columns=['l_xy_r','l_xy_l','r_sdr','l_sdr','t','l_xy_r_visi','l_xy_l_visi'])

    df_upper_arm_length_left=df_u_arm.iloc[:,1]
    df_upper_arm_length_right=df_u_arm.iloc[:,0]

    df_upper_arm_length_left_visi=df_u_arm.iloc[:,-1]
    df_upper_arm_length_right_visi=df_u_arm.iloc[:,-2]

    ##Length of lower arm
    l_arm=[]
    for i in range(len(key_array)):
        t=key_array[i][14][3]
        l_xy_r=lower_arm_r(i)[0]/ubl(i)
        l_xy_l=lower_arm_l(i)[0]/ubl(i)
        l_xy_r_visi=lower_arm_r(i)[1]
        l_xy_l_visi=lower_arm_l(i)[1]


        #l_xy_r=lower_arm_r(i)
        #l_xy_l=lower_arm_l(i)
        right = key_array[i][12][0]
        left = key_array[i][11][0]
        l_arm.append([l_xy_r])
        l_arm[i].append(l_xy_l)
        l_arm[i].append(right)
        l_arm[i].append(left)
        l_arm[i].append(t)
        l_arm[i].append(l_xy_r_visi)
        l_arm[i].append(l_xy_l_visi)
    lower_arm=np.array(l_arm) 
    df_l_arm = pd.DataFrame(lower_arm,columns=['l_xy_r','l_xy_l','r_sdr','l_sdr','t','l_xy_r_visi','l_xy_r_visi'])

    df_lower_arm_length_left=df_l_arm.iloc[:,1]
    df_lower_arm_length_right=df_l_arm.iloc[:,0]
    df_lower_arm_length_left_visi=df_l_arm.iloc[:,-1]
    df_lower_arm_length_right_visi=df_l_arm.iloc[:,-2]

    ##Length of arm
    arm=[]
    for i in range(len(key_array)):
        t=key_array[i][12][3]
        #l_xy_r=upper_arm_r(i)+lower_arm_r(i)+palm_to_finger_r(i)/ubl(i)
        #l_xy_l=upper_arm_l(i)+lower_arm_l(i)+palm_to_finger_l(i)/ubl(i)

        l_xy_r=(upper_arm_r(i)[0]+lower_arm_r(i)[0]+palm_to_finger_r(i)[0])/ubl(i)
        l_xy_l=(upper_arm_l(i)[0]+lower_arm_l(i)[0]+palm_to_finger_l(i)[0])/ubl(i)
        l_xy_r_visi=(upper_arm_r(i)[1]*lower_arm_r(i)[1]*palm_to_finger_r(i)[1])
        l_xy_l_visi=(upper_arm_l(i)[1]*lower_arm_l(i)[1]*palm_to_finger_l(i)[1])

        right = key_array[i][12][0]
        left = key_array[i][11][0]
        arm.append([l_xy_r])
        arm[i].append(l_xy_l)
        arm[i].append(right)
        arm[i].append(left)
        arm[i].append(t)
        arm[i].append(l_xy_r_visi)
        arm[i].append(l_xy_l_visi)
    Length_of_arm=np.array(arm) 
    df_arm = pd.DataFrame(Length_of_arm,columns=['l_xy_r','l_xy_l','r_sdr','l_sdr','t','l_xy_r_visi','l_xy_l_visi'])

    df_arm_length_left=df_arm.iloc[:,1]
    df_arm_length_right=df_arm.iloc[:,0]
    df_arm_length_left_visi=df_arm.iloc[:,-1]
    df_arm_length_right_visi=df_arm.iloc[:,-2]


    ##Length of thigh
    th=[]
    for i in range(len(key_array)):
        t=key_array[i][24][3]
        l_xy_r=thigh_r(i)[0]/ubl(i)
        l_xy_l=thigh_l(i)[0]/ubl(i)
        l_xy_r_visi=thigh_r(i)[1]
        l_xy_l_visi=thigh_l(i)[1]
        #l_xy_r=thigh_r(i)
        #l_xy_l=thigh_l(i)
        right = key_array[i][12][0]
        left = key_array[i][11][0]
        th.append([l_xy_r])
        th[i].append(l_xy_l)
        th[i].append(right)
        th[i].append(left)
        th[i].append(t)
        th[i].append(l_xy_r_visi)
        th[i].append(l_xy_l_visi)
    thigh=np.array(th) 
    df_thigh= pd.DataFrame(thigh,columns=['l_xy_r','l_xy_l','r_sdr','l_sdr','t','l_xy_r_visi','l_xy_l_visi'])

    df_thigh_length_left=df_thigh.iloc[:,1]
    df_thigh_length_right=df_thigh.iloc[:,0]
    df_thigh_length_left_visi=df_thigh.iloc[:,-1]
    df_thigh_length_right_visi=df_thigh.iloc[:,-2]

    ##Length of shank
    shk=[]
    for i in range(len(key_array)):
        t=key_array[i][26][3]
        l_xy_r=shank_r(i)[0]/ubl(i)
        l_xy_l=shank_r(i)[0]/ubl(i)
        l_xy_r_visi=shank_r(i)[1]
        l_xy_l_visi=shank_r(i)[1]

        #l_xy_r=shank_r(i)
        #l_xy_l=shank_l(i)
        right = key_array[i][12][0]
        left = key_array[i][11][0]
        shk.append([l_xy_r])
        shk[i].append(l_xy_l)
        shk[i].append(right)
        shk[i].append(left)
        shk[i].append(t)
        shk[i].append(l_xy_r_visi)
        shk[i].append(l_xy_l_visi)
    shank=np.array(shk) 
    df_shk = pd.DataFrame(shank,columns=['l_xy_r','l_xy_l','r_sdr','l_sdr','t','l_xy_r_visi','l_xy_l_visi'])

    df_shank_length_left=df_shk.iloc[:,1]
    df_shank_length_right=df_shk.iloc[:,0]
    df_shank_length_left_visi=df_shk.iloc[:,-1]
    df_shank_length_right_visi=df_shk.iloc[:,-2]

    ##Face width ear to ear
    Face_width=[]
    for i in range(len(key_array)):
        t=key_array[i][8][3]
        l_xy=face_width(i)[0]/ubl(i)
        l_xy_visi=face_width(i)[1]
        right = key_array[i][12][0]
        left = key_array[i][11][0]
        Face_width.append([l_xy])
        Face_width[i].append(right)
        Face_width[i].append(left)
        Face_width[i].append(t)
        Face_width[i].append(l_xy_visi)
    Face_width_ear_to_ear=np.array(Face_width) 
    df_Face_width = pd.DataFrame(Face_width_ear_to_ear,columns=['l_xy','r_sdr','l_sdr','t','l_xy_visi'])

    df_Face_width_ear_to_ear=df_Face_width.iloc[:,0]
    df_Face_width_ear_to_ear_visi=df_Face_width.iloc[:,-1]

    ##Mouth width
    mouth=[]
    for i in range(len(key_array)):
        t=key_array[i][9][3]
        def mouth_width(k):
            l_xy=length(key_array[k][9][0],key_array[k][10][0],key_array[k][9][1],key_array[k][10][1],0,0)
            visi=key_array[k][9][4]*key_array[k][10][4]
            return l_xy,visi
        l_xy=mouth_width(i)[0]/ubl(i)
        l_xy_visi=mouth_width(i)[1]
        right = key_array[i][12][0]
        left = key_array[i][11][0]
        mouth.append([l_xy])
        mouth[i].append(right)
        mouth[i].append(left)
        mouth[i].append(t)
        mouth[i].append(l_xy_visi)
    mouth_width=np.array(mouth) 
    df_mouth = pd.DataFrame(mouth_width,columns=['l_xy','r_sdr','l_sdr','t','l_xy_visi'])

    df_mouth_width=df_mouth.iloc[:,0]
    df_mouth_width_visi=df_mouth.iloc[:,-1]

    ##eye size
    eye=[]
    for i in range(len(key_array)):
        t=key_array[i][6][3]
        l_xy=eye_size(i)[0]/ubl(i)
        l_xy_visi=eye_size(i)[1]
        right = key_array[i][12][0]
        left = key_array[i][11][0]
        eye.append([l_xy])
        eye[i].append(right)
        eye[i].append(left)
        eye[i].append(t)
        eye[i].append(l_xy_visi)
    eye_width=np.array(eye) 
    df_eye = pd.DataFrame(eye_width,columns=['l_xy','r_sdr','l_sdr','t','l_xy_visi'])

    df_eye_size=df_eye.iloc[:,0]
    df_eye_size_visi=df_eye.iloc[:,-1]

    ##Distance between eyes Midpoints
    eyes_mid=[]
    for i in range(len(key_array)):
        t=key_array[i][5][3]
        l_xy=eyes_midpoints(i)[0]/ubl(i)
        l_xy_visi=eyes_midpoints(i)[1]
        right = key_array[i][12][0]
        left = key_array[i][11][0]
        eyes_mid.append([l_xy])
        eyes_mid[i].append(right)
        eyes_mid[i].append(left)
        eyes_mid[i].append(t)
        eyes_mid[i].append(l_xy_visi)

    eyes_mid_width=np.array(eyes_mid) 
    df_eyes_mid = pd.DataFrame(eyes_mid_width,columns=['l_xy','r_sdr','l_sdr','t','l_xy_visi'])

    df_eyes_mid_width=df_eyes_mid.iloc[:,0]
    df_eyes_mid_width_visi=df_eyes_mid.iloc[:,-1]

    ###DETECTING GAIT
    PARA=[]

    count=0
    for i in range(len(key_array)):
        t=key_array[i][28][3]
        po=math.degrees(pelvic_obliquity(key_array[i][23][0],key_array[i][24][0],key_array[i][23][1],key_array[i][24][1]))
        sw=step_width(key_array[i][11][0],key_array[i][12][0],key_array[i][23][0],key_array[i][24][0],key_array[i][11][1],key_array[i][12][1],key_array[i][23][1],key_array[i][24][1],key_array[i][29][0],key_array[i][30][0])
        flr=foot_lifting(key_array[i][28][1],key_array[i][27][1],key_array[i][11][0],key_array[i][12][0],key_array[i][23][0],key_array[i][24][0],key_array[i][11][1],key_array[i][12][1],key_array[i][23][1],key_array[i][24][1])
        fll=foot_lifting(key_array[i][27][1],key_array[i][28][1],key_array[i][11][0],key_array[i][12][0],key_array[i][23][0],key_array[i][24][0],key_array[i][11][1],key_array[i][12][1],key_array[i][23][1],key_array[i][24][1])
        stl=step_length(key_array[i][11][0],key_array[i][12][0],key_array[i][23][0],key_array[i][24][0],key_array[i][11][1],key_array[i][12][1],key_array[i][23][1],key_array[i][24][1],key_array[i][27][2],key_array[i][28][2])
        count = count +1
        right = key_array[i][12][0]
        left = key_array[i][11][0]

        ##TODO: I comment out this listed {sl}
        #S_L.append([sl])

        PARA.append([t])
        PARA[i].append(count)
        PARA[i].append(po)
        PARA[i].append(sw)
        PARA[i].append(flr)
        PARA[i].append(fll)
        PARA[i].append(stl)
        PARA[i].append(right)
        PARA[i].append(left)

    parameters = np.array(PARA)
    df_para = pd.DataFrame(parameters,columns=['t','count','po','sw','flr','fll','stl','right','left'])
    #print(df_para)

    ##Foot lifting right
    flr_peaks=get_peaks(df_para,df_para.flr,0.2)
    Time_to_index_ratio=(flr_peaks[1]-flr_peaks[0])/(df.t[flr_peaks[1]]-df.t[flr_peaks[0]])
    df_time=time_diff(flr_peaks,df)
    flr_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.flr,0.2,0.1)

    fll_peaks=get_peaks(df_para,df_para.fll,0.2)
    Time_to_index_ratio=(fll_peaks[1]-fll_peaks[0])/(df.t[fll_peaks[1]]-df.t[fll_peaks[0]])
    df_time=time_diff(fll_peaks,df)
    fll_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.fll,0.2,0.1)

    df_lower_body_length=df_lb.iloc[:,0]
    df_shoulder_length=df_shoulder_l.iloc[:,0]
    df_hip_size=df_hip.iloc[:,0]
    df_upper_arm_length_left=df_u_arm.iloc[:,1]
    df_upper_arm_length_right=df_u_arm.iloc[:,0]
    df_lower_arm_length_left=df_l_arm.iloc[:,1]
    df_lower_arm_length_right=df_l_arm.iloc[:,0]
    df_arm_length_left=df_arm.iloc[:,1]
    df_arm_length_right=df_arm.iloc[:,0]
    df_thigh_length_left=df_thigh.iloc[:,1]
    df_thigh_length_right=df_thigh.iloc[:,0]
    df_shank_length_left=df_shk.iloc[:,1]
    df_shank_length_right=df_shk.iloc[:,0]
    df_Face_width_ear_to_ear=df_Face_width.iloc[:,0]
    df_mouth_width=df_mouth.iloc[:,0]
    df_eye_size=df_eye.iloc[:,0]
    df_eyes_mid_width=df_eyes_mid.iloc[:,0]

    df_lower_body_length_avg=calculate_averages(flr_peaks_rd,
                                                df_lower_body_length)
    df_shoulder_length_avg=calculate_averages(flr_peaks_rd,
                                                df_shoulder_length)
    df_hip_size_avg=calculate_averages(flr_peaks_rd,
                                                df_hip_size)
    df_upper_arm_length_left_avg=calculate_averages(flr_peaks_rd,
                                                df_upper_arm_length_left)
    df_upper_arm_length_right_avg=calculate_averages(flr_peaks_rd,
                                                df_upper_arm_length_right)
    df_lower_arm_length_left_avg=calculate_averages(flr_peaks_rd,
                                                df_lower_arm_length_left)
    df_lower_arm_length_right_avg=calculate_averages(flr_peaks_rd,
                                                df_lower_arm_length_right)
    df_arm_length_left_avg=calculate_averages(flr_peaks_rd,
                                                df_arm_length_left)
    df_arm_length_right_avg=calculate_averages(flr_peaks_rd,
                                                df_arm_length_right)
    df_thigh_length_left_avg=calculate_averages(flr_peaks_rd,
                                                df_thigh_length_left)
    df_thigh_length_right_avg=calculate_averages(flr_peaks_rd,
                                                df_thigh_length_right)
    df_shank_length_left_avg=calculate_averages(flr_peaks_rd,
                                                df_shank_length_left)
    df_shank_length_right_avg=calculate_averages(flr_peaks_rd,
                                                df_shank_length_right)
    df_Face_width_ear_to_ear_avg=calculate_averages(flr_peaks_rd,
                                                df_Face_width_ear_to_ear)
    df_mouth_width_avg=calculate_averages(flr_peaks_rd,
                                                df_mouth_width)
    df_eye_size_avg=calculate_averages(flr_peaks_rd,
                                                df_eye_size)
    df_eyes_mid_width_avg=calculate_averages(flr_peaks_rd,
                                                df_eyes_mid_width)

    ##visibility
    lower_body_length_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_lower_body_length_visi)
    shoulder_length_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_shoulder_length_visi)
    hip_size_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_hip_size_visi)
    upper_arm_length_left_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_upper_arm_length_left_visi)
    upper_arm_length_right_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_upper_arm_length_right_visi)
    lower_arm_length_left_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_lower_arm_length_left_visi)
    lower_arm_length_right_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_lower_arm_length_right_visi)
    arm_length_left_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_arm_length_left_visi)
    arm_length_right_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_arm_length_right_visi)
    thigh_length_left_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_thigh_length_left_visi)
    thigh_length_right_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_thigh_length_right_visi)
    shank_length_left_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_shank_length_left_visi)
    shank_length_right_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_shank_length_right_visi)
    Face_width_ear_to_ear_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_Face_width_ear_to_ear_visi)
    mouth_width_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_mouth_width_visi)
    eye_size_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_eye_size_visi)
    eyes_mid_width_avg_visi=calculate_averages_visi(flr_peaks_rd,
                                                df_eyes_mid_width_visi)

    df_lower_body_length_avg_visi=pd.DataFrame(lower_body_length_avg_visi,columns=['lower_body_length_avg_visi'])
    df_shoulder_length_avg_visi=pd.DataFrame(shoulder_length_avg_visi,columns=['shoulder_length_avg_visi'])
    df_hip_size_avg_visi=pd.DataFrame(hip_size_avg_visi,columns=['hip_size_avg_visi'])
    df_upper_arm_length_left_avg_visi=pd.DataFrame(upper_arm_length_left_avg_visi,columns=['upper_arm_length_left_avg_visi'])
    df_upper_arm_length_right_avg_visi=pd.DataFrame(upper_arm_length_right_avg_visi,columns=['upper_arm_length_right_avg_visi'])
    df_lower_arm_length_left_avg_visi=pd.DataFrame(lower_arm_length_left_avg_visi,columns=['lower_arm_length_left_avg_visi'])
    df_lower_arm_length_right_avg_visi=pd.DataFrame(lower_arm_length_right_avg_visi,columns=['lower_arm_length_right_avg_visi'])
    df_arm_length_left_avg_visi=pd.DataFrame(arm_length_left_avg_visi,columns=['arm_length_left_avg_visi'])
    df_arm_length_right_avg_visi=pd.DataFrame(arm_length_right_avg_visi,columns=['arm_length_right_avg_visi'])
    df_thigh_length_left_avg_visi=pd.DataFrame(thigh_length_left_avg_visi,columns=['thigh_length_left_avg_visi'])
    df_thigh_length_right_avg_visi=pd.DataFrame(thigh_length_right_avg_visi,columns=['thigh_length_right_avg_visi'])
    df_shank_length_left_avg_visi=pd.DataFrame(shank_length_left_avg_visi,columns=['shank_length_left_avg_visi'])
    df_shank_length_right_avg_visi=pd.DataFrame(shank_length_right_avg_visi,columns=['shank_length_right_avg_visi'])
    df_Face_width_ear_to_ear_avg_visi=pd.DataFrame(Face_width_ear_to_ear_avg_visi,columns=['sFace_width_ear_to_ear_avg_visi'])
    df_mouth_width_avg_visi=pd.DataFrame(mouth_width_avg_visi,columns=['mouth_width_avg_visi'])
    df_eye_size_avg_visi=pd.DataFrame(eye_size_avg_visi,columns=['eye_size_avg_visi'])
    df_eyes_mid_width_avg_visi=pd.DataFrame(eyes_mid_width_avg_visi,columns=['eyes_mid_width_avg_visi'])

    df_visi = pd.concat([df_lower_body_length_avg_visi,
                                df_shoulder_length_avg_visi,
                                df_hip_size_avg_visi,
                                df_upper_arm_length_left_avg_visi,
                                df_upper_arm_length_right_avg_visi,
                                df_lower_arm_length_left_avg_visi,
                                df_lower_arm_length_right_avg_visi,
                                df_arm_length_left_avg_visi,
                                df_arm_length_right_avg_visi,
                                df_thigh_length_left_avg_visi,
                                df_thigh_length_right_avg_visi,
                                df_shank_length_left_avg_visi,
                                df_shank_length_right_avg_visi,
                                df_Face_width_ear_to_ear_avg_visi,
                                df_mouth_width_avg_visi,
                                df_eye_size_avg_visi,
                                df_eyes_mid_width_avg_visi], axis=1,
                                )


    df_static_front = pd.concat([df_lower_body_length_avg[1],df_lower_body_length_avg[0],
                                df_shoulder_length_avg[0],
                                df_hip_size_avg[0],
                                df_upper_arm_length_left_avg[0],
                                df_upper_arm_length_right_avg[0],
                                df_lower_arm_length_left_avg[0],
                                df_lower_arm_length_right_avg[0],
                                df_arm_length_left_avg[0],
                                df_arm_length_right_avg[0],
                                df_thigh_length_left_avg[0],
                                df_thigh_length_right_avg[0],
                                df_shank_length_left_avg[0],
                                df_shank_length_right_avg[0],
                                df_Face_width_ear_to_ear_avg[0],
                                df_mouth_width_avg[0],
                                df_eye_size_avg[0],
                                df_eyes_mid_width_avg[0]], axis=1,
                                keys=['time','df_lower_body_length_avg',
                                    'df_shoulder_length_avg',
                                    'df_hip_size_avg',
                                    'df_upper_arm_length_left_avg',
                                    'df_upper_arm_length_right_avg',
                                    'df_lower_arm_length_left_avg',
                                    'df_lower_arm_length_right_avg',
                                    'df_arm_length_left_avg',
                                    'df_arm_length_right_avg',
                                    'df_thigh_length_left_avg',
                                    'df_thigh_length_right_avg',
                                    'df_shank_length_left_avg',
                                    'df_shank_length_right_avg',
                                    'df_Face_width_ear_to_ear_avg',
                                    'df_mouth_width_avg',
                                    'df_eye_size_avg',
                                    'df_eyes_mid_width_avg'])

    df_index=['time','df_lower_body_length_avg',
                                    'df_shoulder_length_avg',
                                    'df_hip_size_avg',
                                    'df_upper_arm_length_left_avg',
                                    'df_upper_arm_length_right_avg',
                                    'df_lower_arm_length_left_avg',
                                    'df_lower_arm_length_right_avg',
                                    'df_arm_length_left_avg',
                                    'df_arm_length_right_avg',
                                    'df_thigh_length_left_avg',
                                    'df_thigh_length_right_avg',
                                    'df_shank_length_left_avg',
                                    'df_shank_length_right_avg',
                                    'df_shank_length_left_avg',
                                    'df_Face_width_ear_to_ear_avg',
                                    'df_mouth_width_avg',
                                    'df_eye_size_avg',
                                    'df_eyes_mid_width_avg'] 

    df_comb=pd.concat([df_static_front,df_visi], axis=1)



    output = open('valid_para/Static_front.pkl', 'wb')
    pickle.dump(df_comb, output)
    output.close()
        
        
        
        
        