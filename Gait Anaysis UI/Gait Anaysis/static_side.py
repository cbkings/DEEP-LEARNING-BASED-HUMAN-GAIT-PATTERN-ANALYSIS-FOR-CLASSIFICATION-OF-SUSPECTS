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

def static_side(key_array):
    def length(x1,x2,y1,y2,z1,z2):
        output=(math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2))
        return output

    def thigh_r(k):
        l_xy=length(key_array[k][24][0],key_array[k][26][0],key_array[k][24][1],key_array[k][26][1],0,0)
        visi=(key_array[k][24][4])*key_array[k][26][4]
        return l_xy,visi
    def thigh_l(k):
        l_xy=length(key_array[k][23][0],key_array[k][25][0],key_array[k][23][1],key_array[k][25][1],0,0)
        visi=(key_array[k][23][4])*key_array[k][25][4]
        return l_xy,visi
    def shank_r(k):
        l_xy=length(key_array[k][28][0],key_array[k][26][0],key_array[k][26][1],key_array[k][28][1],0,0)
        visi=(key_array[k][28][4])*key_array[k][26][4]
        return l_xy,visi
    def shank_l(k):
        l_xy=length(key_array[k][27][0],key_array[k][25][0],key_array[k][27][1],key_array[k][25][1],0,0)
        visi=(key_array[k][27][4])*key_array[k][25][4]
        return l_xy,visi
    def ankle_to_heel_r(k):
        l_xy=length(key_array[k][28][0],key_array[k][30][0],key_array[k][28][1],key_array[k][30][1],0,0)
        visi=(key_array[k][28][4])*key_array[k][30][4]
        return l_xy,visi
    def ankle_to_heel_l(k):
        l_xy=length(key_array[k][27][0],key_array[k][29][0],key_array[k][27][1],key_array[k][29][1],0,0)
        visi=(key_array[k][27][4])*key_array[k][29][4]
        return l_xy,visi

    def upper_body(x11,x12,x23,x24,y11,y12,y23,y24,z11,z12,z23,z24):
        output=math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2+(((z11+z12)/2)-((z23+z24)/2))**2)
        return output

    def ubl(k):
        l_xy=upper_body(key_array[k][11][0],key_array[k][12][0],key_array[k][23][0],key_array[k][24][0],key_array[k][11][1],key_array[k][12][1],key_array[k][23][1],key_array[k][24][1],0,0,0,0)
        return l_xy

    def center(k):
        cent=(key_array[k][11][0]+key_array[k][12][0]+key_array[k][23][0]+key_array[k][24][0])/4
        return cent

    def upper_body(x11,x12,x23,x24,y11,y12,y23,y24,z11,z12,z23,z24):
        output=math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2+(((z11+z12)/2)-((z23+z24)/2))**2)
        return output

    def ubl(k):
        l_xy=upper_body(key_array[k][11][0],key_array[k][12][0],key_array[k][23][0],key_array[k][24][0],key_array[k][11][1],key_array[k][12][1],key_array[k][23][1],key_array[k][24][1],0,0,0,0)
        return l_xy

    def step_length(x11,x12,x23,x24,y11,y12,y23,y24,x27,x28):
        step_length=abs(x28 - x27)
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output = step_length/body_length
        return output

    def get_peaks(df,y,p):
        peak, _ = find_peaks(y,prominence=p)
        plt.figure(figsize=(20,5))
        #plt.xlim(0, 100)
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
        #plt.xlim(0, 100)
        #plt.title("Step Length")
        #plt.plot(df.t,y)
        #plt.plot(df.t[peaks], y[peaks],"X",markerfacecolor='red',markersize=8)
        #plt.show()
        #print(peaks)
        return peaks

    def time_diff(peaks, df):
        array=[]
        for i in range(len(peaks)-1):
            td=df.t[peaks[i+1]]-df.t[peaks[i]]
            array.append(td)
        return array

    def length(x1,x2,y1,y2,z1,z2):
        output=(math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2))
        return output
    def upper_arm_r(k):
        l_xy=length(key_array[k][12][0],key_array[k][14][0],key_array[k][12][1],key_array[k][14][1],0,0)
        visi=(key_array[k][12][4])*key_array[k][14][4]
        return l_xy,visi
    def upper_arm_l(k):
        l_xy=length(key_array[k][11][0],key_array[k][13][0],key_array[k][11][1],key_array[k][13][1],0,0)
        visi=(key_array[k][11][4])*key_array[k][13][4]
        #print('upper_arm_l ',visi)
        return l_xy,visi

    def lower_arm_r(k):
        l_xy=length(key_array[k][16][0],key_array[k][14][0],key_array[k][16][1],key_array[k][14][1],0,0)
        visi=(key_array[k][16][4])*key_array[k][14][4]
        return l_xy,visi
    def lower_arm_l(k):
        l_xy=length(key_array[k][15][0],key_array[k][13][0],key_array[k][15][1],key_array[k][13][1],0,0)
        visi=(key_array[k][15][4])*key_array[k][13][4]
        #print('lower_arm_l ',visi)
        return l_xy,visi

    def palm_to_finger_r(k):
        l_xy=length(key_array[k][16][0],key_array[k][20][0],key_array[k][16][1],key_array[k][20][1],0,0)
        visi=(key_array[k][16][4])*key_array[k][20][4]
        return l_xy,visi
    def palm_to_finger_l(k):
        l_xy=length(key_array[k][15][0],key_array[k][19][0],key_array[k][15][1],key_array[k][19][1],0,0)
        visi=(key_array[k][15][4])*key_array[k][19][4]
        #print('palm_to_finger_l ',visi)
        return l_xy,visi

    def foot_length_r(k):
        l_xy=length(key_array[k][30][0],key_array[k][32][0],key_array[k][30][1],key_array[k][32][1],0,0)
        visi=(key_array[k][30][4])*key_array[k][32][4]
        return l_xy,visi
    def foot_length_l(k):
        l_xy=length(key_array[k][29][0],key_array[k][31][0],key_array[k][29][1],key_array[k][31][1],0,0)
        visi=(key_array[k][29][4])*key_array[k][31][4]
        return l_xy,visi

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
        return result,time

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
        return result

    import os

    #f_file = open('valid_key_arrays/key_array_evidance_side.pkl', 'rb')



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

        #arm length
        x_al1,x_al2 = (key_array[i][14][0],key_array[i][12][0])
        y_al1,y_al2 = (key_array[i][14][1],key_array[i][12][1])
        #z_al1,z_al2 = (key_array[i][14][2],key_array[i][12][2])
        al=math.sqrt(((x_al1-x_al2)**2)+((y_al1-y_al2)**2))
        #al=math.sqrt(((x_al1-x_al2)**2)+((y_al1-y_al2)**2)+((z_al1-z_al2)**2))

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
        H[i].append(al)
        H[i].append(sl/ubh)
        H[i].append(al/ubh)


    H_array = np.array(H)
    #print(H_array)

    df = pd.DataFrame(H_array,columns=['ubh','t','center',
                                'r_sdr','l_sdr','sl','al','sln','aln'])
    #print(df)

    PARA=[]
    S_L=[]

    count=0
    for i in range(len(key_array)):
        t=key_array[i][28][3]
        stl=step_length(key_array[i][11][0],key_array[i][12][0],key_array[i][23][0],key_array[i][24][0],key_array[i][11][1],key_array[i][12][1],key_array[i][23][1],key_array[i][24][1],key_array[i][27][0],key_array[i][28][0])
        count = count +1
        right = key_array[i][12][2]
        left = key_array[i][11][2]


        PARA.append([t])
        PARA[i].append(count)
        PARA[i].append(stl)
        PARA[i].append(right)
        PARA[i].append(left)

    parameters = np.array(PARA)
    df_para = pd.DataFrame(parameters,columns=['t','count','step_length','z_right','z_left'])
    #print(df_para)

    p=0.4
    stl_peaks=get_peaks(df_para,df_para.step_length,p)

    Time_to_index_ratio=(stl_peaks[1]-stl_peaks[0])/(df.t[stl_peaks[1]]-df.t[stl_peaks[0]])
    df_time=time_diff(stl_peaks,df)
    stl_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.step_length,p,0.2)
    peaks=stl_peaks_rd

    ##Lower Body length
    array =[]
    for i in range(len(key_array)):
        t=key_array[i][28][3]
        #center of upper body in depth
        center=(key_array[i][11][2]+key_array[i][12][2]+key_array[i][23][2]+key_array[i][24][2])/4
        c=center
        array.append([c])
        array[i].append(t)
    array = np.array(array)
    df_c_t = pd.DataFrame(array,columns=['c','t'])
    #print(df_c_t)

    #df_c_t.plot(x ='t', y='c', kind = 'line',figsize=(20,5))
    #plt.show()

    from scipy.interpolate import UnivariateSpline

    spl = UnivariateSpline(df_c_t.t, df_c_t.c, k=4, s=0)
    diffspl = spl.derivative()

    lb=[]
    for i in range(len(key_array)):
        #print(i)
        t=key_array[i][11][3] 
        #r=thigh_r(i)+shank_r(i)
        #l=thigh_l(i)+shank_l(i)

        r=(thigh_r(i)[0]+shank_r(i)[0])/ubl(i)
        r_visi=thigh_r(i)[1]*shank_r(i)[1]

        l=(thigh_l(i)[0]+shank_l(i)[0])/ubl(i)
        l_visi=thigh_l(i)[1]*shank_l(i)[1]

        leg_ratio=thigh_r(i)[0]/thigh_l(i)[0]
        #c= center(i)
        #center of upper body in depth
        center=(key_array[i][11][2]+key_array[i][12][2]+key_array[i][23][2]+key_array[i][24][2])/4
        c=center
        lb.append([r])
        lb[i].append(l)
        lb[i].append(c)
        lb[i].append(leg_ratio)
        lb[i].append(t)
        lb[i].append(diffspl(t))
        lb[i].append(r_visi)
        lb[i].append(l_visi)

    ratio_ub_lb=np.array(lb)
    df_lb = pd.DataFrame(ratio_ub_lb,columns=['lbl_r','lbl_l','center','leg_ratio','t','velocity','visi_lbl_r','visi_lbl_l'])
    #print(df_lb)

    lbl_r_avg,time_lbl_r_avg=calculate_averages(stl_peaks_rd,df_lb.lbl_r)
    lbl_r_avg_visi=calculate_averages_visi(stl_peaks_rd,df_lb.visi_lbl_r)
    lbl_r_avg=[lbl_r_avg,time_lbl_r_avg]

    lbl_l_avg,time_lbl_l_avg=calculate_averages(stl_peaks_rd,df_lb.lbl_l)
    lbl_l_avg_visi=calculate_averages_visi(stl_peaks_rd,df_lb.visi_lbl_l)
    lbl_l_avg=[lbl_l_avg,time_lbl_l_avg]

    df_lower_body_length_left_avg=pd.DataFrame(lbl_l_avg)
    df_lower_body_length_left_avg=df_lower_body_length_left_avg.transpose()
    df_lower_body_length_left_avg
    df_lower_body_length_right_avg=pd.DataFrame(lbl_r_avg)
    df_lower_body_length_right_avg=df_lower_body_length_right_avg.transpose()
    df_lower_body_length_right_avg

    #plt.figure(figsize=(20,5))
    #plt.plot(df_lb.t,df_lb.lbl_r)
    #plt.plot(df_lb.t[peaks], df_lb.lbl_r[peaks],"X",markerfacecolor='red',markersize=8)

    #plt.plot(df_lower_body_length_right_avg.iloc[:,1],df_lower_body_length_right_avg.iloc[:,0]
    #         ,"X",markerfacecolor='blue',markersize=8)

    #plt.plot(df_lower_body_length_right_avg.iloc[:,1],df_lower_body_length_right_avg.iloc[:,0]
    #         ,color='green',markersize=8)

    #plt.ylim(0, 2)
    #plt.show()

    #plt.figure(figsize=(20,5))
    #plt.ylim(0, 2)
    #plt.plot(df_lower_body_length_right_avg.iloc[:,1],df_lower_body_length_right_avg.iloc[:,0])


    ##Length of upper arm
    u_arm=[]
    for i in range(len(key_array)):
        t=key_array[i][12][3]
        center=(key_array[i][11][2]+key_array[i][12][2]+key_array[i][23][2]+key_array[i][24][2])/4
        c=center
        l_urm_r=upper_arm_r(i)[0]/ubl(i)
        l_urm_l=upper_arm_l(i)[0]/ubl(i)
        l_urm_r_visi=upper_arm_r(i)[1]
        l_urm_l_visi=upper_arm_l(i)[1]

        u_arm.append([l_urm_r])
        u_arm[i].append(l_urm_l)
        u_arm[i].append(c)
        u_arm[i].append(t)
        u_arm[i].append(diffspl(t))
        u_arm[i].append(l_urm_r_visi)
        u_arm[i].append(l_urm_l_visi)

    u_arm= np.array(u_arm)
    df_u_arm = pd.DataFrame(u_arm,columns=['upper_arm_r','upper_arm_l','c','t','velocity','visi_l_urm_r','visi_l_urm_l'])
    #print(df_u_arm)

    u_arm_r_avg,time_u_arm_r_avg=calculate_averages(stl_peaks_rd,df_u_arm.upper_arm_r)
    u_arm_r_avg_visi=calculate_averages_visi(stl_peaks_rd,df_u_arm.visi_l_urm_r)
    u_arm_r_avg=[u_arm_r_avg,time_u_arm_r_avg]

    u_arm_l_avg,time_u_arm_l_avg=calculate_averages(stl_peaks_rd,df_u_arm.upper_arm_l)
    u_arm_l_avg_visi=calculate_averages_visi(stl_peaks_rd,df_u_arm.visi_l_urm_l)
    u_arm_l_avg=[u_arm_l_avg,time_u_arm_l_avg]

    df_upper_arm_length_left_avg=pd.DataFrame(u_arm_l_avg)
    df_upper_arm_length_left_avg=df_upper_arm_length_left_avg.transpose()
    df_upper_arm_length_left_avg
    df_upper_arm_length_right_avg=pd.DataFrame(u_arm_r_avg)
    df_upper_arm_length_right_avg=df_upper_arm_length_right_avg.transpose()
    df_upper_arm_length_right_avg

    #plt.figure(figsize=(20,5))
    #plt.plot(df_u_arm.t,df_u_arm.upper_arm_r)
    #plt.plot(df_u_arm.t[peaks], df_u_arm.upper_arm_r[peaks],"X",markerfacecolor='red',markersize=8)

    #plt.plot(df_upper_arm_length_right_avg.iloc[:,1],df_upper_arm_length_right_avg.iloc[:,0]
    #         ,"X",markerfacecolor='blue',markersize=8)

    #plt.ylim(0, 1)
    #plt.show()

    #plt.figure(figsize=(20,5))
    #plt.plot(df_upper_arm_length_right_avg.iloc[:,1],df_upper_arm_length_right_avg.iloc[:,0])
    #plt.ylim(0, 1)

    ##Length of Lower arm
    l_arm=[]
    for i in range(len(key_array)):
        t=key_array[i][14][3]
        center=(key_array[i][11][2]+key_array[i][12][2]+key_array[i][23][2]+key_array[i][24][2])/4
        c=center
        la_r=lower_arm_r(i)[0]/ubl(i)
        la_l=lower_arm_l(i)[0]/ubl(i)
        la_r_visi=lower_arm_r(i)[1]
        la_l_visi=lower_arm_l(i)[1]

        l_arm.append([la_r])
        l_arm[i].append(la_l)
        l_arm[i].append(c)
        l_arm[i].append(t)
        l_arm[i].append(diffspl(t))
        l_arm[i].append(la_r_visi)
        l_arm[i].append(la_l_visi)

    lwr_arm= np.array(l_arm)
    df_larm = pd.DataFrame(lwr_arm,columns=['r','l','c','t','velocity','visi_larm_r','visi_larm_l'])
    #print(df_larm)
    #df_larm.plot(x ='t', kind = 'line')
    #plt.show()

    larm_r_avg,time_larm_r_avg=calculate_averages(stl_peaks_rd,df_larm.r)
    larm_r_avg_visi=calculate_averages_visi(stl_peaks_rd,df_larm.visi_larm_r)
    larm_r_avg=[larm_r_avg,time_larm_r_avg]

    larm_l_avg,time_larm_l_avg=calculate_averages(stl_peaks_rd,df_larm.l)
    larm_l_avg_visi=calculate_averages_visi(stl_peaks_rd,df_larm.visi_larm_l)
    larm_l_avg=[larm_l_avg,time_larm_l_avg]

    df_lower_arm_length_left_avg=pd.DataFrame(larm_l_avg)
    df_lower_arm_length_left_avg=df_lower_arm_length_left_avg.transpose()
    df_lower_arm_length_left_avg
    df_lower_arm_length_right_avg=pd.DataFrame(larm_r_avg)
    df_lower_arm_length_right_avg=df_lower_arm_length_right_avg.transpose()
    df_lower_arm_length_right_avg

    #plt.figure(figsize=(20,5))
    #plt.plot(df_larm.t,df_larm.r)
    #plt.plot(df_larm.t[peaks], df_larm.r[peaks],"X",markerfacecolor='red',markersize=8)

    #plt.plot(df_lower_arm_length_right_avg.iloc[:,1],df_lower_arm_length_right_avg.iloc[:,0]
    #         ,"X",markerfacecolor='blue',markersize=8)

    #plt.ylim(0, 1)
    #plt.show()

    #plt.figure(figsize=(20,5))
    #plt.plot(df_lower_arm_length_right_avg.iloc[:,1],df_lower_arm_length_right_avg.iloc[:,0])
    #plt.ylim(0, 1)

    ##Length of arm
    Length_of_arm=[]
    for i in range(len(key_array)):
        t=key_array[i][12][3]
        l_r=upper_arm_r(i)[0]+lower_arm_r(i)[0]+palm_to_finger_r(i)[0]/ubl(i)
        l_l=upper_arm_l(i)[0]+lower_arm_l(i)[0]+palm_to_finger_l(i)[0]/ubl(i)
        l_r_visi=upper_arm_r(i)[1]*lower_arm_r(i)[1]*palm_to_finger_r(i)[1]
        l_l_visi=upper_arm_l(i)[1]*lower_arm_l(i)[1]*palm_to_finger_l(i)[1]
        #print('l_l_visi ',l_l_visi)

        center=(key_array[i][11][2]+key_array[i][12][2]+key_array[i][23][2]+key_array[i][24][2])/4
        c=center
        Length_of_arm.append([l_r])
        Length_of_arm[i].append(l_l)
        Length_of_arm[i].append(c)
        Length_of_arm[i].append(t)
        Length_of_arm[i].append(diffspl(t))
        Length_of_arm[i].append(l_r_visi)
        Length_of_arm[i].append(l_l_visi)

    Length_of_arm= np.array(Length_of_arm)
    df_arm = pd.DataFrame(Length_of_arm,columns=['r','l','c','t','velocity','visi_l_r','visi_l_l'])
    #print(df_arm)

    arm_r_avg,time_arm_r_avg=calculate_averages(stl_peaks_rd,df_arm.r)
    arm_r_avg_visi=calculate_averages_visi(stl_peaks_rd,df_arm.visi_l_r)
    arm_r_avg=[arm_r_avg,time_arm_r_avg]

    arm_l_avg,time_arm_l_avg=calculate_averages(stl_peaks_rd,df_arm.l)
    arm_l_avg_visi=calculate_averages_visi(stl_peaks_rd,df_arm.visi_l_l)
    arm_l_avg=[arm_l_avg,time_arm_l_avg]

    df_arm_length_left_avg=pd.DataFrame(arm_l_avg)
    df_arm_length_left_avg=df_arm_length_left_avg.transpose()
    df_arm_length_left_avg
    df_arm_length_right_avg=pd.DataFrame(arm_r_avg)
    df_arm_length_right_avg=df_arm_length_right_avg.transpose()
    df_arm_length_right_avg

    #plt.figure(figsize=(20,5))
    #plt.plot(df_arm.t,df_arm.r)
    #plt.plot(df_arm.t[peaks], df_arm.r[peaks],"X",markerfacecolor='red',markersize=8)

    #plt.plot(df_arm_length_right_avg.iloc[:,1],df_arm_length_right_avg.iloc[:,0]
    #         ,"X",markerfacecolor='blue',markersize=8)

    #plt.plot(df_arm_length_right_avg.iloc[:,1],df_arm_length_right_avg.iloc[:,0],color='green')

    #plt.ylim(0, 1)
    #plt.show()


    #plt.figure(figsize=(20,5))
    #plt.plot(df_arm_length_right_avg.iloc[:,1],df_arm_length_right_avg.iloc[:,0])
    #plt.ylim(0, 1)


    ##Length of Thigh
    th_r=[]
    for i in range(len(key_array)):
        t=key_array[i][24][3]
        r=thigh_r(i)[0]/ubl(i)
        l=thigh_l(i)[0]/ubl(i)
        r_visi=thigh_r(i)[1]
        l_visi=thigh_l(i)[1]

        center=(key_array[i][11][2]+key_array[i][12][2]+key_array[i][23][2]+key_array[i][24][2])/4
        c=center
        leg_ratio=shank_r(i)[0]/shank_l(i)[0]
        th_r.append([r])
        th_r[i].append(l)
        th_r[i].append(leg_ratio)
        th_r[i].append(c)
        th_r[i].append(t)
        th_r[i].append(diffspl(t))
        th_r[i].append(r_visi)
        th_r[i].append(l_visi)

    th_r= np.array(th_r)
    df_thigh = pd.DataFrame(th_r,columns=['r','l','leg_ratio','c','t','velocity','visi_thigh_r','visi_thigh_l'])
    #print(df_thigh)

    thigh_r_avg,time_thigh_r_avg=calculate_averages(stl_peaks_rd,df_thigh.r)
    thigh_r_avg_visi=calculate_averages_visi(stl_peaks_rd,df_thigh.visi_thigh_r)
    thigh_r_avg=[thigh_r_avg,time_thigh_r_avg]

    thigh_l_avg,time_thigh_l_avg=calculate_averages(stl_peaks_rd,df_thigh.l)
    thigh_l_avg_visi=calculate_averages_visi(stl_peaks_rd,df_thigh.visi_thigh_l)
    thigh_l_avg=[thigh_l_avg,time_thigh_l_avg]

    df_thigh_length_left_avg=pd.DataFrame(thigh_l_avg)
    df_thigh_length_left_avg=df_thigh_length_left_avg.transpose()
    df_thigh_length_left_avg

    df_thigh_length_right_avg=pd.DataFrame(thigh_r_avg)
    df_thigh_length_right_avg=df_thigh_length_right_avg.transpose()
    df_thigh_length_right_avg

    #plt.figure(figsize=(20,5))
    #plt.plot(df_thigh.t,df_thigh.r)
    #plt.plot(df_thigh.t[peaks], df_thigh.r[peaks],"X",markerfacecolor='red',markersize=8)

    #plt.plot(df_thigh_length_right_avg.iloc[:,1],df_thigh_length_right_avg.iloc[:,0]
    #         ,"X",markerfacecolor='blue',markersize=8)

    #plt.plot(df_thigh_length_right_avg.iloc[:,1],df_thigh_length_right_avg.iloc[:,0],color='green')

    #plt.ylim(0, 1)
    #plt.show()

    #plt.figure(figsize=(20,5))
    #plt.plot(df_thigh_length_right_avg.iloc[:,1],df_thigh_length_right_avg.iloc[:,0])
    #plt.ylim(0, 1)

    ##Length of shank
    shk=[]
    for i in range(len(key_array)):
        t=key_array[i][26][3]
        r=shank_r(i)[0]/ubl(i)
        l=shank_l(i)[0]/ubl(i)
        r_visi=shank_r(i)[1]
        l_visi=shank_l(i)[1]
        center=(key_array[i][11][2]+key_array[i][12][2]+key_array[i][23][2]+key_array[i][24][2])/4
        c=center
        leg_ratio=thigh_r(i)[0]/thigh_l(i)[0]
        shk.append([r])
        shk[i].append(l)
        shk[i].append(leg_ratio)
        shk[i].append(c)
        shk[i].append(t)
        shk[i].append(diffspl(t))
        shk[i].append(r_visi)
        shk[i].append(l_visi)

    shk= np.array(shk)
    df_shank = pd.DataFrame(shk,columns=['r','l','leg_ratio','c','t','velocity','visi_shk_r','visi_shk_l'])
    #print(df_shank)

    shank_r_avg,time_shank_r_avg=calculate_averages(stl_peaks_rd,df_shank.r)
    shank_r_avg_visi=calculate_averages_visi(stl_peaks_rd,df_shank.visi_shk_r)
    shank_r_avg=[shank_r_avg,time_shank_r_avg]

    shank_l_avg,time_shank_l_avg=calculate_averages(stl_peaks_rd,df_shank.l)
    shank_l_avg_visi=calculate_averages_visi(stl_peaks_rd,df_shank.visi_shk_l)
    shank_l_avg=[shank_l_avg,time_shank_l_avg]

    df_shank_length_left_avg=pd.DataFrame(shank_l_avg)
    df_shank_length_left_avg=df_shank_length_left_avg.transpose()

    df_shank_length_right_avg=pd.DataFrame(shank_r_avg)
    df_shank_length_right_avg=df_shank_length_right_avg.transpose()

    #plt.figure(figsize=(20,5))
    #plt.title("Shank Length right")
    #plt.plot(df_shank.t,df_shank.r)
    #plt.plot(df_shank.t[peaks], df_shank.r[peaks],"X",markerfacecolor='red',markersize=8)

    #plt.plot(df_shank_length_right_avg.iloc[:,1],df_shank_length_right_avg.iloc[:,0]
    #         ,"X",markerfacecolor='blue',markersize=8)

    #plt.plot(df_shank_length_right_avg.iloc[:,1],df_shank_length_right_avg.iloc[:,0],color='green')

    #plt.ylim(0, 1)
    #plt.show()

    #plt.figure(figsize=(20,5))
    #plt.plot(df_shank_length_right_avg.iloc[:,1],df_shank_length_right_avg.iloc[:,0])
    #plt.ylim(0, 1)


    ##Foot length
    foot=[]
    for i in range(len(key_array)):
        t=key_array[i][30][3]
        r=foot_length_r(i)[0]/ubl(i)
        l=foot_length_l(i)[0]/ubl(i)
        r_visi=foot_length_r(i)[1]
        l_visi=foot_length_l(i)[1]
        leg_ratio=thigh_r(i)[0]/thigh_l(i)[0]
        center=(key_array[i][11][2]+key_array[i][12][2]+key_array[i][23][2]+key_array[i][24][2])/4
        c=center
        foot.append([r])
        foot[i].append(l)
        foot[i].append(leg_ratio)
        foot[i].append(c)
        foot[i].append(t)
        foot[i].append(diffspl(t))
        foot[i].append(r_visi)
        foot[i].append(l_visi)
    foot= np.array(foot)
    df_foot = pd.DataFrame(foot,columns=['foot_r','foot_l','leg_ratio','c','t','velocity','visi_foot_r','visi_foot_l'])
    #print(df_foot)

    foot_r_avg,time_foot_r_avg=calculate_averages(stl_peaks_rd,df_foot.foot_r)
    foot_r_avg_visi=calculate_averages_visi(stl_peaks_rd,df_foot.visi_foot_r)
    foot_r_avg=[foot_r_avg,time_foot_r_avg]

    foot_l_avg,time_foot_l_avg=calculate_averages(stl_peaks_rd,df_foot.foot_l)
    foot_l_avg_visi=calculate_averages_visi(stl_peaks_rd,df_foot.visi_foot_l)
    foot_l_avg=[foot_l_avg,time_foot_l_avg]
    df_foot_length_left_avg=pd.DataFrame(foot_l_avg)
    df_foot_length_left_avg=df_foot_length_left_avg.transpose()
    df_foot_length_right_avg=pd.DataFrame(foot_r_avg)
    df_foot_length_right_avg=df_foot_length_right_avg.transpose()

    #plt.figure(figsize=(20,5))
    #plt.plot(df_foot.t,df_foot.foot_r)
    #plt.plot(df_foot.t[peaks], df_foot.foot_r[peaks],"X",markerfacecolor='red',markersize=8)

    #plt.plot(df_foot_length_right_avg.iloc[:,1],df_foot_length_right_avg.iloc[:,0]
    #         ,"X",markerfacecolor='blue',markersize=8)

    #plt.plot(df_foot_length_right_avg.iloc[:,1],df_foot_length_right_avg.iloc[:,0]
    #         ,color='green',markersize=8)

    #plt.ylim(0, 1)
    #plt.show()

    #plt.figure(figsize=(20,5))
    #plt.plot(df_foot_length_right_avg.iloc[:,1],df_foot_length_right_avg.iloc[:,0])
    #plt.ylim(0, 0.5)

    df_lbl_r_avg_visi=pd.DataFrame(lbl_r_avg_visi,columns=['lbl_r_avg_visi'])
    df_lbl_l_avg_visi=pd.DataFrame(lbl_l_avg_visi,columns=['lbl_l_avg_visi'])
    df_u_arm_r_avg_visi=pd.DataFrame(u_arm_r_avg_visi,columns=['u_arm_r_avg_visi'])
    df_u_arm_l_avg_visi=pd.DataFrame(u_arm_l_avg_visi,columns=['u_arm_l_avg_visi'])
    df_larm_r_avg_visi=pd.DataFrame(larm_r_avg_visi,columns=['larm_r_avg_visi'])
    df_larm_l_avg_visi=pd.DataFrame(larm_l_avg_visi,columns=['larm_l_avg_visi'])
    df_arm_r_avg_visi=pd.DataFrame(arm_r_avg_visi,columns=['arm_r_avg_visi'])
    df_arm_l_avg_visi=pd.DataFrame(arm_l_avg_visi,columns=['arm_l_avg_visi'])
    df_thigh_r_avg_visi=pd.DataFrame(thigh_r_avg_visi,columns=['thigh_r_avg_visi'])
    df_thigh_l_avg_visi=pd.DataFrame(thigh_l_avg_visi,columns=['thigh_l_avg_visi'])
    df_shank_r_avg_visi=pd.DataFrame(shank_r_avg_visi,columns=['shank_r_avg_visi'])
    df_shank_l_avg_visi=pd.DataFrame(shank_l_avg_visi,columns=['shank_l_avg_visi'])
    df_foot_r_avg_visi=pd.DataFrame(foot_r_avg_visi,columns=['foot_r_avg_visi'])
    df_foot_l_avg_visi=pd.DataFrame(foot_l_avg_visi,columns=['foot_l_avg_visi'])


    df_visi=pd.concat([df_lbl_r_avg_visi,
                    df_lbl_l_avg_visi,
                    df_u_arm_r_avg_visi,
                    df_u_arm_l_avg_visi,
                    df_larm_r_avg_visi,
                    df_larm_l_avg_visi,
                    df_arm_r_avg_visi,
                    df_arm_l_avg_visi,
                    df_thigh_r_avg_visi,
                    df_thigh_l_avg_visi,
                    df_shank_r_avg_visi,
                    df_shank_l_avg_visi,
                    df_foot_r_avg_visi,
                    df_foot_l_avg_visi], axis=1,
                    )

    df_static_side = pd.concat([df_lower_body_length_left_avg[1],
                                df_lower_body_length_left_avg[0],
                                df_lower_body_length_right_avg[0],
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
                                df_foot_length_left_avg[0],
                                df_foot_length_right_avg[0]], axis=1, 
                                keys=['time','lower_body_left_avg',
                                    'lower_body_right_avg',
                                    'upper_arm_left_avg',
                                    'upper_arm_right_avg',
                                    'lower_arm_left_avg',
                                    'lower_arm_right_avg',
                                    'arm_left_avg',
                                    'arm_right_avg',
                                    'thigh_left_avg',
                                    'thigh_right_avg',
                                    'shank_left_avg',
                                    'shank_right_avg',
                                    'foot_left_avg',
                                    'foot_right_avg',])

    df_comb=pd.concat([df_static_side,df_visi], axis=1)


    output = open('valid_para/Static_side.pkl', 'wb')
    pickle.dump(df_comb, output)
    output.close()

            #print(fps,cam_angle,cond,name,direction,vid_num)
