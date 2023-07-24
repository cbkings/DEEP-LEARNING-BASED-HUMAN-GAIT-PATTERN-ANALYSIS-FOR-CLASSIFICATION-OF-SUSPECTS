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
from scipy.stats import binom
from scipy import signal

def dynamic_front(key_array):
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

    def pelvic_obliquity(x23,x24,y23,y24):
        output=math.atan((abs(y23-y24))/(abs(x23-x24)))
        return output

    def shoulder_length(x11,x12,x23,x24,y11,y12,y23,y24):
        x_sl1,x_sl2 = (key_array[i][11][0],key_array[i][12][0])
        y_sl1,y_sl2 = (key_array[i][11][1],key_array[i][12][1])
        #z_sl1,z_sl2 = (key_array[i][11][2],key_array[i][12][2])
        sl=math.sqrt(((x_sl1-x_sl2)**2)+((y_sl1-y_sl2)**2))
        shoulder_length=sl
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output = shoulder_length/body_length
        #output = shoulder_len
        return output

    def step_width(x11,x12,x23,x24,y11,y12,y23,y24,x29,x30):
        step_width=abs(x30 - x29)
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output = step_width/body_length
        #output = step_wid
        return output

    def foot_lifting(y28,y27,x11,x12,x23,x24,y11,y12,y23,y24):
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output=(y28 - y27)/body_length
        return output

    def hip_ankle_right(y28,x11,x12,x23,x24,y11,y12,y23,y24):
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output = abs(y24-y28)/body_length
        return output

    def hip_ankle_left(y27,x11,x12,x23,x24,y11,y12,y23,y24):
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output = abs(y23-y27)/body_length
        return output

    def upl(x11,x12,x23,x24,y11,y12,y23,y24):
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        return body_length

    def step_length(x11,x12,x23,x24,y11,y12,y23,y24,z27,z28):
        step_length=abs(z28 - z27)
        body_length = math.sqrt((((x11+x12)/2)-((x23+x24)/2))**2+(((y11+y12)/2)-((y23+y24)/2))**2)
        output = step_length/body_length
        #output = step_len
        return output

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

    def shoulder_length_visi(x11,x12):
        output = x11*x12
        #output = shoulder_len
        return output

    #f_file = open('valid_key_arrays/key_array_evidance_front.pkl', 'rb')
    #print(f_file)



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
    #print(df)

    ##Parameter calculation
    PARA=[]

    count=0
    for i in range(len(key_array)):
        t=key_array[i][28][3]
        sl=shoulder_length(key_array[i][11][0],key_array[i][12][0],key_array[i][23][0],key_array[i][24][0],key_array[i][11][1],key_array[i][12][1],key_array[i][23][1],key_array[i][24][1])
        po=math.degrees(pelvic_obliquity(key_array[i][23][0],key_array[i][24][0],key_array[i][23][1],key_array[i][24][1]))
        sw=step_width(key_array[i][11][0],key_array[i][12][0],key_array[i][23][0],key_array[i][24][0],key_array[i][11][1],key_array[i][12][1],key_array[i][23][1],key_array[i][24][1],key_array[i][29][0],key_array[i][30][0])
        flr=foot_lifting(key_array[i][28][1],key_array[i][27][1],key_array[i][11][0],key_array[i][12][0],key_array[i][23][0],key_array[i][24][0],key_array[i][11][1],key_array[i][12][1],key_array[i][23][1],key_array[i][24][1])
        fll=foot_lifting(key_array[i][27][1],key_array[i][28][1],key_array[i][11][0],key_array[i][12][0],key_array[i][23][0],key_array[i][24][0],key_array[i][11][1],key_array[i][12][1],key_array[i][23][1],key_array[i][24][1])
        hakr=hip_ankle_right(key_array[i][28][1],key_array[i][11][0],key_array[i][12][0],key_array[i][23][0],key_array[i][24][0],key_array[i][11][1],key_array[i][12][1],key_array[i][23][1],key_array[i][24][1])
        hakl=hip_ankle_left(key_array[i][27][1],key_array[i][11][0],key_array[i][12][0],key_array[i][23][0],key_array[i][24][0],key_array[i][11][1],key_array[i][12][1],key_array[i][23][1],key_array[i][24][1])
        stl=step_length(key_array[i][11][0],key_array[i][12][0],key_array[i][23][0],key_array[i][24][0],key_array[i][11][1],key_array[i][12][1],key_array[i][23][1],key_array[i][24][1],key_array[i][27][2],key_array[i][28][2])
        count = count +1
        right = key_array[i][12][0]
        left = key_array[i][11][0]

        #visibility
        sl_visi=shoulder_length_visi(key_array[i][11][4],key_array[i][12][4])
        po_visi=pelvic_obliquity_visi(key_array[i][23][4],key_array[i][24][4])
        sw_visi=step_width_visi(key_array[i][29][4],key_array[i][30][4])
        flr_visi=foot_lifting_visi(key_array[i][28][4],key_array[i][27][4])
        fll_visi=foot_lifting_visi(key_array[i][27][4],key_array[i][28][4])
        hakr_visi=hip_ankle_right_visi(key_array[i][28][4],key_array[i][24][4])
        hakl_visi=hip_ankle_left_visi(key_array[i][27][4],key_array[i][23][4])
        stl_visi=step_length_visi(key_array[i][27][4],key_array[i][28][4])


        PARA.append([t])
        PARA[i].append(count)
        PARA[i].append(sl)
        PARA[i].append(po)
        PARA[i].append(sw)
        PARA[i].append(flr)
        PARA[i].append(fll)
        PARA[i].append(hakr)
        PARA[i].append(hakl)
        PARA[i].append(stl)
        PARA[i].append(right)
        PARA[i].append(left)

        PARA[i].append(sl_visi)
        PARA[i].append(stl_visi)
        PARA[i].append(hakr_visi)
        PARA[i].append(hakl_visi)
        PARA[i].append(flr_visi)
        PARA[i].append(fll_visi)
        PARA[i].append(po_visi)
        PARA[i].append(sw_visi)

    parameters = np.array(PARA)
    df_para = pd.DataFrame(parameters,columns=['t','count','sl','po','sw','flr','fll','hakr','hakl','stl','right','left','sl_visi','stl_visi',
                                            'hakr_visi','hakl_visi','flr_visi','fll_visi','po_visi','sw_visi'])
    #print(df_para)

    #Pelvic obliquity
    po_peaks=get_peaks(df_para,df_para.po,0.01)
    Time_to_index_ratio=(po_peaks[1]-po_peaks[0])/(df.t[po_peaks[1]]-df.t[po_peaks[0]])
    df_time=time_diff(po_peaks,df)
    po_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.po,0.01,1)
    po_peaks
    po_peaks_rd


    #Step Width
    sw_peaks=get_peaks(df_para,df_para.sw,0.001)
    Time_to_index_ratio=(sw_peaks[1]-sw_peaks[0])/(df.t[sw_peaks[1]]-df.t[sw_peaks[0]])
    df_time=time_diff(sw_peaks,df)
    sw_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.sw,0.01,0.5)

    #Foot lifting right
    flr_peaks=get_peaks(df_para,df_para.flr,0.2)
    Time_to_index_ratio=(flr_peaks[1]-flr_peaks[0])/(df.t[flr_peaks[1]]-df.t[flr_peaks[0]])
    df_time=time_diff(flr_peaks,df)
    #print("DETECTING GAIT___Foot lifting right")
    flr_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.flr,0.2,0.1)


    #Foot Lifting Left
    fll_peaks=get_peaks(df_para,df_para.fll,0.2)
    Time_to_index_ratio=(fll_peaks[1]-fll_peaks[0])/(df.t[fll_peaks[1]]-df.t[fll_peaks[0]])
    df_time=time_diff(fll_peaks,df)
    fll_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.fll,0.2,0.1)

    #Hip To Ankle Right
    hakr_valleys=get_valleys(df_para,df_para.hakr,0.01)
    Time_to_index_ratio=(hakr_valleys[1]-hakr_valleys[0])/(df.t[hakr_valleys[1]]-df.t[hakr_valleys[0]])
    df_time=time_diff(hakr_valleys,df)
    hakr_valleys_rd=remove_distinct_valleys(df_time,df,Time_to_index_ratio,df_para.hakr,0.01,2)

    #Hip To Ankle Light
    hakl_valleys=get_valleys(df_para,df_para.hakl,0.001)
    Time_to_index_ratio=(hakl_valleys[1]-hakl_valleys[0])/(df.t[hakl_valleys[1]]-df.t[hakl_valleys[0]])
    df_time=time_diff(hakl_valleys,df)
    hakl_valleys_rd=remove_distinct_valleys(df_time,df,Time_to_index_ratio,df_para.hakl,0.01,2)

    #Step Length
    stl_peaks=get_peaks(df_para,df_para.stl,0.1)
    Time_to_index_ratio=(stl_peaks[1]-stl_peaks[0])/(df.t[stl_peaks[1]]-df.t[stl_peaks[0]])
    df_time=time_diff(stl_peaks,df)
    stl_peaks_rd=remove_distinct_peaks(df_time,df,Time_to_index_ratio,df_para.stl,0.2,0.4)



    D_f = [[None],[None], [None], [None], [None], [None], [None], [None],
        [None], [None], [None], [None], [None], [None], [None]]

    t_time=[]
    stl=[]
    hakr=[]
    hakl=[]
    flr=[]
    fll=[]
    po=[]
    sw=[]

    stl_visi=[]
    hakr_visi=[]
    hakl_visi=[]
    flr_visi=[]
    fll_visi=[]
    po_visi=[]
    sw_visi=[]

    #flr_peaks_rd

    for i in range(len(flr_peaks_rd)-1):
        start_index = flr_peaks_rd[i]
        end_index = flr_peaks_rd[i+1]
        t=df_para.t[start_index]
        flr_data = [df_para.flr[start_index],df_para.flr[end_index]]
        flr_para=np.mean(flr_data)
        t_time.append(t)
        flr.append(flr_para)
        hakr.append(para_valleys_value(df_para.hakr,hakr_valleys_rd,'hakr',start_index,end_index))
        hakl.append(para_valleys_value(df_para.hakl,hakl_valleys_rd,'hakl',start_index,end_index))
        fll.append(para_peaks_value(df_para.fll,fll_peaks_rd,'fll',start_index,end_index))
        po.append(para_peaks_value(df_para.po,po_peaks_rd,'po',start_index,end_index))
        sw.append(para_peaks_value(df_para.sw,sw_peaks_rd,'sw',start_index,end_index))
        stl.append(para_peaks_value(df_para.stl,stl_peaks_rd,'stl',start_index,end_index))

        #visi
        flr_data_visi = [df_para.flr_visi[start_index],df_para.flr_visi[end_index]]
        flr_para_visi=np.mean(flr_data_visi)
        flr_visi.append(flr_para_visi)
        hakr_visi.append(para_valleys_value(df_para.hakr_visi,hakr_valleys_rd,'hakr_visi',start_index,end_index))
        hakl_visi.append(para_valleys_value(df_para.hakl_visi,hakl_valleys_rd,'hakl_visi',start_index,end_index))
        fll_visi.append(para_peaks_value(df_para.fll_visi,fll_peaks_rd,'fll_visi',start_index,end_index))
        po_visi.append(para_peaks_value(df_para.po_visi,po_peaks_rd,'po_visi',start_index,end_index))
        sw_visi.append(para_peaks_value(df_para.sw_visi,sw_peaks_rd,'sw_visi',start_index,end_index))
        stl_visi.append(para_peaks_value(df_para.stl_visi,stl_peaks_rd,'stl_visi',start_index,end_index))

    D_f[0]=t_time
    D_f[1]=stl
    D_f[2]=hakr
    D_f[3]=hakl
    D_f[4]=flr
    D_f[5]=fll
    D_f[6]=po
    D_f[7]=sw

    D_f[8]=stl_visi
    D_f[9]=hakr_visi
    D_f[10]=hakl_visi
    D_f[11]=flr_visi
    D_f[12]=fll_visi
    D_f[13]=po_visi
    D_f[14]=sw_visi



    para_instant =[]

    count=0
    i=0
    for i in range(len(D_f[i+1])):
        para_instant.append([count])
        count+=1
        for j in range(len(D_f)):
            #print(j,i)
            para_instant[i].append(D_f[j][i])

    para_instant_arry = np.array(para_instant)
    df_para_instant = pd.DataFrame(para_instant_arry,columns=['count','time','stl','hakr','hakl','flr','fll','po','sw',
                                                            'visi_stl','visi_hakr','visi_hakl','visi_flr','visi_fll','visi_po','visi_sw'])


    #output = open('parameters/Dynamic/{}/{}/{}_{}_{}.pkl'.format(cam,direction,name,direction,vid_no), 'wb')
    output = open('valid_para/Dynamic_front.pkl', 'wb')
    pickle.dump(df_para_instant, output)
    output.close()

        