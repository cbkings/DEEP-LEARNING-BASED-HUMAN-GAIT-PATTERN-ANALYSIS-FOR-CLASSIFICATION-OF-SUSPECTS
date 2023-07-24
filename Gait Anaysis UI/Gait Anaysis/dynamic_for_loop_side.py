import mediapipe as mp
import cv2
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy.misc import derivative
from scipy.signal import find_peaks
from statistics import stdev,mean
from scipy.interpolate import interp1d
import itertools as itr
import statistics
import pprint, pickle
import os
#Parameter definitions

def dynamic_side(key_array):
    def hip_angle_right(x26,x24,y24,y26):
        output=math.atan(((x26-x24))/(abs(y24-y26)))
        return output

    def hip_angle_left(x25,x23,y23,y25):
        output=math.atan(((x25-x23))/abs(y23-y25))
        return output

    def hip_ankle_right(y28,x11,x12,x23,x24,y11,y12,y23,y24):
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output = abs(y24-y28)/body_length
        return output

    def hip_ankle_left(y27,x11,x12,x23,x24,y11,y12,y23,y24):
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output = abs(y23-y27)/body_length
        return output

    def thigh_angle(x24,x26,x25,x23,y24,y26,y23,y25):
        a=(math.atan((x26-x24))/abs(y24-y26))
        b=(math.atan((x25-x23))/abs(y23-y25))
        return abs(a-b)

    def knee_angle_xyz(x24,x26,x28,y24,y26,y28,z24,z26,z28):
        a = math.sqrt((x24 -x28)**2 + (y24-y28)**2 + (z24-z28)**2)
        b = math.sqrt((x24 -x26)**2 + (y24-y26)**2 + (z24-z26)**2)
        c = math.sqrt((x26 -x28)**2 + (y26-y28)**2 + (z26-z28)**2) 
        output_xyz= math.acos((b**2+ c**2 - a**2) / (2*b*c))    
        return output_xyz

    def knee_angle_xy(x24,x26,x28,y24,y26,y28):
        p = math.sqrt((x24 -x28)**2 + (y24-y28)**2)
        q = math.sqrt((x24 -x26)**2 + (y24-y26)**2 )
        r = math.sqrt((x26 -x28)**2 + (y26-y28)**2 )
        output_xy= math.acos((q**2+ r**2 - p**2) / (2*q*r))
        return output_xy

    def Shoulder_angle(x14,x12,y12,y14):
        output=math.atan((x14-x12)/((y12-y14)))
        return output

    def foot_lifting(y28,y27,x11,x12,x23,x24,y11,y12,y23,y24):
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output=(y28 - y27)/body_length
        return output

    def thigh_ratio(x24,x26,x23,x25,y24,y26,y23,y25):
        thigh_r = math.sqrt((x24-x26)**2 + (y24-y26)**2)
        thigh_l = math.sqrt((x23-x25)**2 + (y23-y25)**2)
        ratio = thigh_l/thigh_r
        return ratio

    def distance(x11,x12,x31):
        output = (x31 - (x11+x12)/2)
        return output   


    def hipa(y24,y28):
        output=abs(y24-y28)
        return output

    def pelvic_obliquity(x23,x24,y23,y24):
        output=math.atan((abs(y23-y24))/(abs(x23-x24)))
        return output

    def step_width(x11,x12,x23,x24,y11,y12,y23,y24,z29,z30):
        step_width=abs(z30 - z29)
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output = step_width/body_length
        return output

    def step_length(x11,x12,x23,x24,y11,y12,y23,y24,x27,x28):
        step_length=abs(x28 - x27)
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output = step_length/body_length
        return output


    def upbody_length(x11,x12,x23,x24,y11,y12,y23,y24):
        output = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        return output

    def get_peaks(df,y,p):
        peak, _ = find_peaks(y,prominence=p)
        #plt.figure(figsize=(20,5))
        #plt.xlim(0, 100)
        #plt.ylim(0, 60)
        #plt.plot(df.t,y)
        #plt.plot(df.t[peak], y[peak], "X",markerfacecolor='red',markersize=8)
        #plt.show()
        return peak

    def get_valleys(df,y,p):
        valley, _ = find_peaks(-y,prominence=p)
        #plt.figure(figsize=(20,5))
        #plt.plot(df.t,y)
        #plt.plot(df.t[valley], y[valley],"X",markerfacecolor='red',markersize=8)
        #plt.show()
        return valley

    def remove_distinct_peaks(array,df,Time_to_index_ratio,y,p,m):
        modified_mean= mean(array)
        Distance=Time_to_index_ratio*modified_mean*m
        
        peaks, _ = find_peaks(y,prominence=p,distance=Distance)
        #plt.figure(figsize=(20,5))
        #plt.xlim(0, 100)
        #plt.plot(df.t,y)
        #plt.plot(df.t[peaks], y[peaks],"X",markerfacecolor='red',markersize=8)
        #plt.show()
        return peaks

    def remove_distinct_valleys(array,df,Time_to_index_ratio,y,p,m):
        modified_mean= mean(array)
        Distance=Time_to_index_ratio*modified_mean*m
        
        valleys, _ = find_peaks(-y,prominence=p,distance=Distance)
        #plt.figure(figsize=(20,5))
        #plt.plot(df.t,y)
        #plt.plot(df.t[valleys], y[valleys],"X",markerfacecolor='red',markersize=8)
        #plt.show()
        return valleys

    def time_diff(peaks, df):
        array=[]
        for i in range(len(peaks)-1):
            td=df.t[peaks[i+1]]-df.t[peaks[i]]
            array.append(td)
        return array


    def hip_angle_right_visi(v26,v24):
        visi=v26*v24
        return visi

    def hip_angle_left_visi(x25,x23):
        output=x25*x23
        return output

    def hip_ankle_right_visi(y28,y24):
        output = abs(y24*y28)
        return output

    def hip_ankle_left_visi(y27,y23):
        output = abs(y23*y27)
        return output

    def thigh_angle_visi(x24,x26,x25,x23):
        visi=x24*x26*x25*x23
        return visi

    def knee_angle_xyz_visi(x24,x26,x28):
        output_xyz= x24*x26*x28
        return output_xyz

    def knee_angle_xy_visi(x24,x26,x28):
        output_xy= x24*x26*x28
        return output_xy

    def Shoulder_angle_visi(x14,x12):
        output=x14*x12
        return output

    def foot_lifting_visi(y28,y27):
        output=(y28*y27)
        return output

    def thigh_ratio_visi(x24,x26,x23,x25):
        ratio = x24*x26*x23*x25
        return ratio

    def pelvic_obliquity_visi(x23,x24):
        output=x23*x24
        return output

    def step_width_visi(z29,z30):
        step_width=abs(z30 - z29)
        output = abs(z30*z29)
        return output

    def step_length_visi(x27,x28):
        step_length=abs(x28 - x27)
        output = abs(x28*x27)
        return output

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

        #right upper arm length
        x_al1,x_al2 = (key_array[i][14][0],key_array[i][12][0])
        y_al1,y_al2 = (key_array[i][14][1],key_array[i][12][1])
        #z_al1,z_al2 = (key_array[i][14][2],key_array[i][12][2])
        rual=math.sqrt(((x_al1-x_al2)**2)+((y_al1-y_al2)**2))
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
        H[i].append(rual)
        H[i].append(sl/ubh)
        H[i].append(rual/ubh)


    H_array = np.array(H)
    #print(H_array)

    df = pd.DataFrame(H_array,columns=['ubh','t','center','r_sdr','l_sdr','sl','rual','sln','rualn'])

    D_f = [[None],[None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None],
        [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None], [None]]

    PARA=[]

    count=0
    for i in range(len(key_array)):
        t=key_array[i][28][3]

        x11,x12,x14,x23,x24,x25,x26,x27,x28,x29,x30=(key_array[i][11][0],key_array[i][12][0],key_array[i][14][0],key_array[i][23][0],key_array[i][24][0],key_array[i][25][0],key_array[i][26][0],key_array[i][27][0],key_array[i][28][0],key_array[i][29][0],key_array[i][30][0])
        y11,y12,y14,y23,y24,y25,y26,y27,y28=(key_array[i][11][1],key_array[i][12][1],key_array[i][14][1],key_array[i][23][1],key_array[i][24][1],key_array[i][25][1],key_array[i][26][1],key_array[i][27][1],key_array[i][28][1])
        z24,z26,z28,z29,z30=(key_array[i][24][2],key_array[i][26][2],key_array[i][28][2],key_array[i][29][2],key_array[i][30][2])

        v11,v12,v14,v23,v24,v25,v26,v27,v28,v29,v30=(key_array[i][11][4],key_array[i][12][4],key_array[i][14][4],key_array[i][23][4],key_array[i][24][4],key_array[i][25][4],key_array[i][26][4],key_array[i][27][4],key_array[i][28][4],key_array[i][29][4],key_array[i][30][4])


        stl=step_length(x11,x12,x23,x24,y11,y12,y23,y24,x27,x28)
        har=hip_angle_right(x26,x24,y24,y26)
        hal=hip_angle_left(x25,x23,y23,y25)
        hakr=hip_ankle_right(y28,x11,x12,x23,x24,y11,y12,y23,y24)
        hakl=hip_ankle_left(y27,x11,x12,x23,x24,y11,y12,y23,y24)
        ta=thigh_angle(x24,x26,x25,x23,y24,y26,y23,y25)
        kar=knee_angle_xy(x24,x26,x28,y24,y26,y28)
        kal=knee_angle_xy(x23,x25,x27,y23,y25,y27)
        sa=Shoulder_angle(x14,x12,y12,y14)
        flr=foot_lifting(y28,y27,x11,x12,x23,x24,y11,y12,y23,y24)
        fll=foot_lifting(y27,y28,x11,x12,x23,x24,y11,y12,y23,y24)
        tr=thigh_ratio(x24,x26,x23,x25,y24,y26,y23,y25)
        po=pelvic_obliquity(x23,x24,y23,y24)
        sw=step_width(x11,x12,x23,x24,y11,y12,y23,y24,z29,z30)
        count = count +1
        right = key_array[i][12][2]
        left = key_array[i][11][2]

        #visibility
        stl_visi=step_length_visi(v27,v28)
        har_visi=hip_angle_right_visi(v26,v24)
        hal_visi=hip_angle_left_visi(v25,v23)
        hakr_visi=hip_ankle_right_visi(v28,v24)
        hakl_visi=hip_ankle_left_visi(v27,v23)
        ta_visi=thigh_angle_visi(v24,v26,v25,v23)
        kar_visi=knee_angle_xy_visi(v24,v26,v28)
        kal_visi=knee_angle_xy_visi(v23,v25,v27)
        sa_visi=Shoulder_angle_visi(v14,v12)
        flr_visi=foot_lifting_visi(v28,v27)
        fll_visi=foot_lifting_visi(v27,v28)
        tr_visi=thigh_ratio_visi(v24,v26,v23,v25)
        po_visi=pelvic_obliquity_visi(v23,v24)
        sw_visi=step_width_visi(v29,v30)



        PARA.append([t])
        PARA[i].append(count)
        PARA[i].append(stl)
        PARA[i].append(har)
        PARA[i].append(hal)
        PARA[i].append(hakr)
        PARA[i].append(hakl)
        PARA[i].append(ta)
        PARA[i].append(kar)
        PARA[i].append(kal)
        PARA[i].append(sa)
        PARA[i].append(flr)
        PARA[i].append(fll)
        PARA[i].append(tr)
        PARA[i].append(po)
        PARA[i].append(sw)
        PARA[i].append(right)
        PARA[i].append(left)

        PARA[i].append(stl_visi)
        PARA[i].append(har_visi)
        PARA[i].append(hal_visi)
        PARA[i].append(hakr_visi)
        PARA[i].append(hakl_visi)
        PARA[i].append(ta_visi)
        PARA[i].append(kar_visi)
        PARA[i].append(kal_visi)
        PARA[i].append(sa_visi)
        PARA[i].append(flr_visi)
        PARA[i].append(fll_visi)
        PARA[i].append(tr_visi)
        PARA[i].append(po_visi)
        PARA[i].append(sw_visi)

    parameters = np.array(PARA)
    df_para = pd.DataFrame(parameters,columns=['t','count','stl','har','hal','hakr','hakl','ta','kar','kal','sa','flr','fll','tr','po','sw','right','left','visi_stl','visi_har','visi_hal','visi_hakr','visi_hakl','visi_ta','visi_kar','visi_kal','visi_sa','visi_flr','visi_fll','visi_tr','visi_po','visi_sw'])
    #print("df_para")
    #print(df_para)
    def para_peaks_value(para,rd,st,start_index,end_index):
        arr = np.array(rd)
        selected_numbers = arr[np.where((arr >= start_index) & (arr <= end_index))]
        #print(selected_numbers)
        para_val=[]
        for j in selected_numbers:
            para_val.append(para[j])
        
        if len(para_val) == 0:
            max_value = df_para[st][start_index:end_index+1:].max()
            #print('max_value',max_value)
            return max_value
        else:
            average = np.mean(para_val)
            return average
            
            
    def para_valleys_value(para,rd,st,start_index,end_index):
        arr = np.array(rd)
        selected_numbers = arr[np.where((arr >= start_index) & (arr <= end_index))]
        para_val=[]
        for j in selected_numbers:
            para_val.append(para[j])
        
        if len(para_val) == 0:
            min_value = df_para[st][start_index:end_index+1:].min()
            return min_value
        else:
            average = np.mean(para_val)
            return average
    ##Step Length
    stl_peaks=get_peaks(df_para,df_para.stl,0.06)
    Time_to_index_ratio=(stl_peaks[1]-stl_peaks[0])/(df.t[stl_peaks[1]]-df.t[stl_peaks[0]])
    df_time=time_diff(stl_peaks,df)
    stl_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.stl,0.4,0.2)

    stl_peaks_rd

    ##Hip Angle Right
    har_peaks=get_peaks(df_para,df_para.har,0.06)
    Time_to_index_ratio=(har_peaks[1]-har_peaks[0])/(df.t[har_peaks[1]]-df.t[har_peaks[0]])
    df_time=time_diff(har_peaks,df)
    har_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.har,0.06,0.5)

    ##Hip Angle Left
    hal_peaks=get_peaks(df_para,df_para.hal,0.06)
    Time_to_index_ratio=(hal_peaks[1]-hal_peaks[0])/(df.t[hal_peaks[1]]-df.t[hal_peaks[0]])
    df_time=time_diff(hal_peaks,df)
    hal_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.hal,0.06,0.5)

    ##Hip Ankle Right
    hakr_valleys=get_valleys(df_para,df_para.hakr,0.01)
    Time_to_index_ratio=(hakr_valleys[1]-hakr_valleys[0])/(df.t[hakr_valleys[1]]-df.t[hakr_valleys[0]])
    df_time=time_diff(hakr_valleys,df)
    hakr_valleys_rd=remove_distinct_valleys(df_time,df,Time_to_index_ratio,df_para.hakr,0.01,1.5)

    ##Hip Ankle Left
    hakl_valleys=get_valleys(df_para,df_para.hakl,0.001)
    Time_to_index_ratio=(hakl_valleys[1]-hakl_valleys[0])/(df.t[hakl_valleys[1]]-df.t[hakl_valleys[0]])
    df_time=time_diff(hakl_valleys,df)
    hakl_valleys_rd=remove_distinct_valleys(df_time,df,Time_to_index_ratio,df_para.hakl,0.01,2)

    ##Thigh Angle
    ta_peaks=get_peaks(df_para,df_para.ta,0.06)
    Time_to_index_ratio=(ta_peaks[1]-ta_peaks[0])/(df.t[ta_peaks[1]]-df.t[ta_peaks[0]])
    df_time=time_diff(ta_peaks,df)
    ta_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.ta,0.06,0.2)

    ##Knee Angle Right
    #df_para.plot(x ='t', y='kar', kind = 'line',figsize=(20,5))
    #plt.ylim(0,3.2)
    #plt.show()
    kar_valleys=get_valleys(df_para,df_para.kar,0.1)
    Time_to_index_ratio=(kar_valleys[1]-kar_valleys[0])/(df.t[kar_valleys[1]]-df.t[kar_valleys[0]])
    df_time=time_diff(kar_valleys,df)
    kar_valleys_rd=remove_distinct_valleys(df_time,df,Time_to_index_ratio,df_para.kar,0.1,0.5)

    ##Knee Angle Left
    kal_valleys=get_valleys(df_para,df_para.kal,0.05)
    Time_to_index_ratio=(kal_valleys[1]-kal_valleys[0])/(df.t[kal_valleys[1]]-df.t[kal_valleys[0]])
    df_time=time_diff(kal_valleys,df)
    kal_valleys_rd=remove_distinct_valleys(df_time,df,Time_to_index_ratio,df_para.kal,0.05,1)


    ##Shoulder Angle
    sa_peaks=get_peaks(df_para,df_para.sa,0.06)
    Time_to_index_ratio=(sa_peaks[1]-sa_peaks[0])/(df.t[sa_peaks[1]]-df.t[sa_peaks[0]])
    df_time=time_diff(sa_peaks,df)
    sa_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.sa,0.06,0.3)

    ##Foot Lifting Right
    flr_peaks=get_peaks(df_para,df_para.flr,0.01)
    Time_to_index_ratio=(flr_peaks[1]-flr_peaks[0])/(df.t[flr_peaks[1]]-df.t[flr_peaks[0]])
    df_time=time_diff(flr_peaks,df)
    flr_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.flr,0.01,2)

    ##Foot Lifting Left
    fll_peaks=get_peaks(df_para,df_para.fll,0.01)
    Time_to_index_ratio=(fll_peaks[1]-fll_peaks[0])/(df.t[fll_peaks[1]]-df.t[fll_peaks[0]])
    df_time=time_diff(fll_peaks,df)
    fll_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.fll,0.01,2)

    ##Thigh Ratio
    tr_peaks=get_peaks(df_para,df_para.tr,0.01)
    Time_to_index_ratio=(tr_peaks[1]-tr_peaks[0])/(df.t[tr_peaks[1]]-df.t[tr_peaks[0]])
    df_time=time_diff(tr_peaks,df)
    tr_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.tr,0.01,1.6)

    ##Pelvic Obliquity
    po_peaks=get_peaks(df_para,df_para.po,0.01)
    Time_to_index_ratio=(po_peaks[1]-po_peaks[0])/(df.t[po_peaks[1]]-df.t[po_peaks[0]])
    df_time=time_diff(po_peaks,df)
    po_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.po,0.01,2)

    ##Step Width
    sw_peaks=get_peaks(df_para,df_para.sw,0.01)
    Time_to_index_ratio=(sw_peaks[1]-sw_peaks[0])/(df.t[sw_peaks[1]]-df.t[sw_peaks[0]])
    df_time=time_diff(sw_peaks,df)
    sw_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.sw,0.01,2)

    t_time=[]
    stl=[]
    har=[]
    hal=[]
    hakr=[]
    hakl=[]
    ta = []
    kar=[]
    kal=[]
    sa=[]
    flr=[]
    fll=[]
    tr=[]
    po=[]
    sw=[]

    stl_visi=[]
    har_visi=[]
    hal_visi=[]
    hakr_visi=[]
    hakl_visi=[]
    ta_visi = []
    kar_visi=[]
    kal_visi=[]
    sa_visi=[]
    flr_visi=[]
    fll_visi=[]
    tr_visi=[]
    po_visi=[]
    sw_visi=[]


    start_index = stl_peaks_rd[0]
    end_index = stl_peaks_rd[1]
    #print(para_peaks_value(df_para.flr,flr_peaks_rd,'flr',start_index,end_index))


    for i in range(len(stl_peaks_rd)-1):
        start_index = stl_peaks_rd[i]
        end_index = stl_peaks_rd[i+1]
        t=df_para.t[start_index]
        stl_data = [df_para.stl[start_index],df_para.stl[end_index]]
        stl_para=np.mean(stl_data)
        stl.append(stl_para)
        t_time.append(t)
        har.append(para_peaks_value(df_para.har,har_peaks_rd,'har',start_index,end_index))
        hal.append(para_peaks_value(df_para.hal,hal_peaks_rd,'hal',start_index,end_index))
        hakr.append(para_valleys_value(df_para.hakr,hakr_valleys_rd,'hakr',start_index,end_index))
        hakl.append(para_valleys_value(df_para.hakl,hakl_valleys_rd,'hakl',start_index,end_index))
        ta.append(para_peaks_value(df_para.ta,ta_peaks_rd,'ta',start_index,end_index))
        kar.append(para_valleys_value(df_para.kar,kar_valleys_rd,'kar',start_index,end_index))
        kal.append(para_valleys_value(df_para.kar,kar_valleys_rd,'kar',start_index,end_index))
        sa.append(para_peaks_value(df_para.sa,sa_peaks_rd,'sa',start_index,end_index))
        flr.append(para_peaks_value(df_para.flr,flr_peaks_rd,'flr',start_index,end_index))
        fll.append(para_peaks_value(df_para.fll,fll_peaks_rd,'fll',start_index,end_index))
        tr.append(para_peaks_value(df_para.tr,tr_peaks_rd,'tr',start_index,end_index))
        po.append(para_peaks_value(df_para.po,po_peaks_rd,'po',start_index,end_index))
        sw.append(para_peaks_value(df_para.sw,sw_peaks_rd,'sw',start_index,end_index))


        #visi
        stl_data_visi = [df_para.visi_stl[start_index],df_para.visi_stl[end_index]]
        stl_para_visi=np.mean(stl_data_visi)
        stl_visi.append(stl_para_visi)
        har_visi.append(para_peaks_value(df_para.visi_har,har_peaks_rd,'visi_har',start_index,end_index))
        hal_visi.append(para_peaks_value(df_para.visi_hal,hal_peaks_rd,'visi_hal',start_index,end_index))
        hakr_visi.append(para_valleys_value(df_para.visi_hakr,hakr_valleys_rd,'visi_hakr',start_index,end_index))
        hakl_visi.append(para_valleys_value(df_para.visi_hakl,hakl_valleys_rd,'visi_hakl',start_index,end_index))
        ta_visi.append(para_peaks_value(df_para.visi_ta,ta_peaks_rd,'visi_ta',start_index,end_index))
        kar_visi.append(para_valleys_value(df_para.visi_kar,kar_valleys_rd,'visi_kar',start_index,end_index))
        kal_visi.append(para_valleys_value(df_para.visi_kar,kar_valleys_rd,'visi_kar',start_index,end_index))
        sa_visi.append(para_peaks_value(df_para.visi_sa,sa_peaks_rd,'visi_sa',start_index,end_index))
        flr_visi.append(para_peaks_value(df_para.visi_flr,flr_peaks_rd,'visi_flr',start_index,end_index))
        fll_visi.append(para_peaks_value(df_para.visi_fll,fll_peaks_rd,'visi_fll',start_index,end_index))
        tr_visi.append(para_peaks_value(df_para.visi_tr,tr_peaks_rd,'visi_tr',start_index,end_index))
        po_visi.append(para_peaks_value(df_para.visi_po,po_peaks_rd,'visi_po',start_index,end_index))
        sw_visi.append(para_peaks_value(df_para.visi_sw,sw_peaks_rd,'visi_sw',start_index,end_index))


    D_f[0]=t_time
    D_f[1]=stl
    D_f[2]=har
    D_f[3]=hal
    D_f[4]=hakr
    D_f[5]=hakl
    D_f[6]=ta
    D_f[7]=kar
    D_f[8]=kal
    D_f[9]=sa
    D_f[10]=flr
    D_f[11]=fll
    D_f[12]=tr
    D_f[13]=po
    D_f[14]=sw

    D_f[15]=stl_visi
    D_f[16]=har_visi
    D_f[17]=hal_visi
    D_f[18]=hakr_visi
    D_f[19]=hakl_visi
    D_f[20]=ta_visi
    D_f[21]=kar_visi
    D_f[22]=kal_visi
    D_f[23]=sa_visi
    D_f[24]=flr_visi
    D_f[25]=fll_visi
    D_f[26]=tr_visi
    D_f[27]=po_visi
    D_f[28]=sw_visi

    para_instant =[]
    #columns=['time','count','stl','har','hal','hakr','hakl','ta','kar','kal','sa','flr','fll','tr','po','sw']

    count=0
    i=0
    for i in range(len(D_f[i+1])):
        para_instant.append([count])
        count+=1
        for j in range(len(D_f)):
            para_instant[i].append(D_f[j][i])

    para_instant_arry = np.array(para_instant)
    df_para_instant = pd.DataFrame(para_instant_arry,columns=['count','time','stl','har','hal','hakr','hakl','ta','kar','kal','sa','flr','fll','tr','po','sw',
                                                            'visi_stl','visi_har','visi_hal','visi_hakr','visi_hakl','visi_ta','visi_kar','visi_kal','visi_sa','visi_flr','visi_fll','visi_tr','visi_po','visi_sw'])


    output = open('valid_para/Dynamic_side.pkl', 'wb')

    pickle.dump(df_para_instant, output)
    output.close()
    #print(df_para_instant)                             
    #print(fps,cam_angle,cond,name,direction,vid_num)
    #print("   ")
    
