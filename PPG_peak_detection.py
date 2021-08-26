#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import statistics
import heartpy as hp
import neurokit2 as nk
import math


# In[2]:


peaks = np.load("C:/Users/kikaze/Desktop/Data generator/up sampled data/upsample Annotations.npy", allow_pickle=True)
signals = np.load("C:/Users/kikaze/Desktop/Data generator/up sampled data/upsample Raw PPG.npy", allow_pickle=True)
subjects = np.load("C:/Users/kikaze/Desktop/Data generator/up sampled data/upsample signal_Labels.npy", allow_pickle=True)

noise = np.load("C:/Users/kikaze/Desktop/Data generator/up sampled data/upsample_Noise_Signal.npy", allow_pickle=True)
noise_subj = np.load("C:/Users/kikaze/Desktop/Data generator/up sampled data/upsample_Noise_Labels.npy", allow_pickle=True)


# ## Visualize data
# 

# In[3]:


l, c = np.unique(np.concatenate(peaks), return_counts=True)

plt.figure(figsize=(10,7))
plt.plot(c)


# In[4]:


plt.figure(figsize=(15,7))
s = 25
plt.plot(signals[s])
plt.scatter(x=peaks[s], y=signals[s][peaks[s]], color='red')


# In[5]:


plt.plot(noise[0])
# plt.plot(butter_filtering(noise[0], 20, [0.5, 3.5], 5,'bandpass'))


# ## Train test split

# In[6]:


np.random.seed(7)
test_subj = np.random.choice(np.unique(subjects), size=10, replace=False)
train_subj = np.array([i for i in np.unique(subjects) if i not in test_subj])


# In[7]:


test_subj


# In[8]:


train_subj


# In[9]:



test_subj_noise = np.random.choice(np.unique(noise_subj), size=4, replace=False)
train_subj_noise = np.array([i for i in np.unique(noise_subj) if i not in test_subj_noise])


# In[10]:


test_subj_noise


# In[11]:


train_subj_noise


# In[12]:


train_signals = signals[np.in1d(subjects, train_subj)]
train_noise = noise[np.in1d(noise_subj, train_subj_noise)]
train_peaks = peaks[np.in1d(subjects, train_subj)]

test_signals = signals[np.in1d(subjects, test_subj)]
test_noise = noise[np.in1d(noise_subj, test_subj_noise)]
test_peaks = peaks[np.in1d(subjects, test_subj)]


# ## Create training data generator

# In[13]:


def normalizer(arr):
    return 2 * ((arr - arr.min()) / (arr.max() - arr.min())) - 1


# In[14]:


def normalize(arr):
    return 2 * ((arr - arr.min()) / (arr.max() - arr.min()))


# In[15]:


from scipy.signal import butter, filtfilt
from scipy import stats

def butter_filtering(sig,fs,fc,order,btype): 
    #Parameters: signal, sampling frequency, cutoff frequencies, order of the filter, filter type (e.g., bandpass)
    #Returns: filtered signal
    w = np.array(fc)/(fs/2)
    b, a = butter(order, w, btype =btype, analog=False)
    filtered = filtfilt(b, a, sig)
    return(filtered)


# In[16]:


def add_noise(noise_signals, downlimit, uplimit):
    # 1. Randomly select one noise signal
    # 2. Randomly select frame from noise signal
    # 3. Multiply noise with some random scalar
    # Return noise
    random_noise_idx = np.random.randint(0, len(noise_signals))
    random_noise = noise_signals[random_noise_idx]
    noise = random_noise*round(random.uniform(downlimit, uplimit), 5)
    
    return noise


# In[17]:


def SNR_checking(X, y, SNR, SNR_up, SNR_down):
    
    desire_X = []
    desire_y = []
    desire_SNR = []
    
    for i in range(len(SNR)):
        if ((SNR[i] > SNR_down) and (SNR[i] <= SNR_up)):
            
            desire_X.append(X[i])
            desire_y.append(y[i])
            desire_SNR.append(SNR[i])

            
    desire_X = np.asarray(desire_X)
    desire_y = np.asarray(desire_y)
    desire_SNR = np.asarray(desire_SNR)
            
    return desire_X, desire_y, desire_SNR
    


# In[18]:


def Data_extraction(X_inp, y_inp, SNR_inp, min_len):
    
    X_out = []
    y_out = []
    SNR_out = []
    for i in range(len(X_inp)):
        
        X_out.extend(X_inp[i][:min_len])
        y_out.extend(y_inp[i][:min_len])
        SNR_out.extend(SNR_inp[i][:min_len])
        
    return X_out, y_out, SNR_out


# In[19]:


def ppg_generator(signals, peaks, noise_signals, win_s, sampling_rate, batch_size, return_SNR=False):
    win_size = win_s * sampling_rate
    
    # Number of possible peaks, HR = 200
    num_possible_peaks = int(win_s * (200/60))
    
    while True:

        uplimit_noise =   [8.15,  4.88,  2.69, 1.50, 0.87, 0.53, 0.35, 0.22,  0.12, 0.065, 0.040, 0.020, 0.013, 0.004]
        downlimit_noise = [7.95,  4.65,  2.48, 1.43, 0.79, 0.42, 0.24, 0.14, 0.070, 0.045, 0.024, 0.017, 0.008, 0.00]
        SNR_range_up =   [-17.5, -12.5,  -7.5, -2.5,  2.5,  7.5, 12.5, 17.5,  22.5,  27.5,  32.5,  37.5,  42.5,  87.5]
        SNR_range_down = [-22.5, -17.5 ,-12.5, -7.5, -2.5,  2.5,  7.5, 12.5,  17.5,  22.5,  27.5,  32.5,  37.5,  42.5]
        tot_min = []
        tot_X = []
        tot_y = []
        tot_SNR = []
        
        for i in range(len(downlimit_noise)):
            downlimit = downlimit_noise[i]
            uplimit = uplimit_noise[i]
            down_range = SNR_range_down[i]
            up_range = SNR_range_up[i]
            
            
            X_int = []
            y_int = []
            SNR_int = []

            while len(X_int) < batch_size:
                random_sig_idx = np.random.randint(0, len(signals))
                random_sig = signals[random_sig_idx]
                p4sig = peaks[random_sig_idx]

                # Select one window
                beg = np.random.randint(random_sig.shape[0]-win_size)
                end = beg + win_size

                p_in_win = p4sig[(p4sig >= beg) & (p4sig < end)] - beg

                # Check that there is at least one peak in the window and that amount of peaks is natural
                if (p_in_win.shape[0] >= 1) & (p_in_win.shape[0] <= num_possible_peaks):
                    

                    # ------------- Add noise into training example----------
                    noise = add_noise(noise_signals, downlimit, uplimit)
                    frame = random_sig[beg:end]
                    noise_frame = noise[beg:end]
                    frame_noisy = frame + noise_frame
                    
                    if (p_in_win[-1] <= len(frame_noisy)-4):
                    ####### Filtering thr frame #########
                        filtered_frame = butter_filtering(frame_noisy, sampling_rate, 0.6, 5,'highpass')

                        X_int.append(filtered_frame)
                        ############ Calculating the SNR ###################
                        power_sig = np.trapz(frame**2)/(frame.size/100)
                        power_noise = np.trapz(noise_frame**2)/(noise_frame.size/100)

                        Ratio = 10*np.log10(power_sig/power_noise)

                        SNR_int.append(Ratio)


                        labels = np.zeros(win_size)
                        np.put(labels, [p_in_win-2, p_in_win-1, p_in_win, p_in_win+1, p_in_win+2], [1]);

                        y_int.append(labels)
                    
            X_check, y_check, SNR_check = SNR_checking(X_int, y_int, SNR_int, up_range, down_range)
            
            min_check = len(X_check)
            tot_min.append(min_check)
            tot_X.append(X_check)
            tot_y.append(y_check)
            tot_SNR.append(SNR_check)



            
        X, y, SNR = Data_extraction(tot_X, tot_y, tot_SNR, min(tot_min))
        X = np.asarray(X)
        y = np.asarray(y)
        SNR = np.asarray(SNR)


        
        X, y, SNR = shuffle(X, y, SNR,  random_state=0)

        X = np.asarray(X)
        y = np.asarray(y)
        SNR = np.asarray(SNR)


        X = X.reshape(X.shape[0], X.shape[1], 1)
        X = np.apply_along_axis(normalizer, 1, X)
        
        output = (X, y)
        if return_SNR == True:
            output = (X, y, SNR)
        yield output


# In[20]:


train_gen = ppg_generator(
    signals=train_signals,
    peaks=train_peaks,
    noise_signals=train_noise,
    win_s=15,
    sampling_rate=100,
    batch_size=100, return_SNR=False
)



test_gen = ppg_generator(
    signals=test_signals,
    peaks=test_peaks,
    noise_signals=test_noise,
    win_s=15,
    sampling_rate=100,
    batch_size=2000, return_SNR=True
)


# ## importing the model

# ## fit model into the training data 
# 
# Tests with dilated CNN (like wavenet, temporal convolutions TCN)

# In[23]:


iput = tf.keras.layers.Input([1500, 1])
# x = tf.keras.layers.GaussianNoise(stddev=1)(iput)
x = Conv1D(kernel_size=3, filters=4, activation="elu", dilation_rate=1, padding='same')(iput)
x = Conv1D(kernel_size=3, filters=8, activation="elu", dilation_rate=2, padding='same')(x)
x = Conv1D(kernel_size=3, filters=8, activation="elu", dilation_rate=4, padding='same')(x)
x = Conv1D(kernel_size=3, filters=16, activation="elu", dilation_rate=8, padding='same')(x)
x = Conv1D(kernel_size=3, filters=16, activation="elu", dilation_rate=16, padding='same')(x)
x = Conv1D(kernel_size=3, filters=32, activation="elu", dilation_rate=32, padding='same')(x)
oput = Conv1D(kernel_size=3, filters=1, activation="sigmoid", dilation_rate=64, padding='same')(x)

model = tf.keras.Model(inputs=iput, outputs=oput)

model.compile(
    optimizer='adam',
    loss = 'binary_crossentropy',
    metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(train_gen,
                    steps_per_epoch=60,
                    epochs=200)


# In[24]:


train_X, train_y = next(train_gen)
yhat = model.predict(train_X)


# ## Using the Reconstructed model

# In[21]:


from tensorflow import keras
reconstructed_model = keras.models.load_model('trained_model')


# In[25]:


test_X, test_y, SNR = next(test_gen)
ypred = reconstructed_model.predict(test_X)
test_eval = reconstructed_model.evaluate(test_X, test_y, verbose=1)


# In[26]:


############# Finding the TP and FP and FN for CNN  ############
def metrics(true_binary_label, achieved_label, binary_achieved_label):
    FP = []
    FN = []
    TP = []

    for i in range(len(achieved_label)):
        
        if true_binary_label[achieved_label[i]] == 1:
            TP.append(1)
            
        elif true_binary_label[achieved_label[i]] == 0:
            FP.append(1)
            
    j = 0
    while(j<len(true_binary_label)-1):
        
        if true_binary_label[j] == 1:
            p = j
            
            while true_binary_label[p]== 1 and p<len(true_binary_label)-1:
                
                p += 1

            test = true_binary_label[j:p]
 
            if len(test)>0:
                flag = []
                for l in range(len(test)):
                    
                    if binary_achieved_label[l+j] == 1:
                        flag.append(1)
                    else:
                        flag.append(0)
      
                if sum(flag) == 0:
                    FN.append(1)
                else:
                    FN.append(0)
            if len(test)>0:
                j=j + len(test)
            elif len(test)==0:
                j +=1
            
        else:
            j += 1
                
    return TP, FP, FN
                
        
        
            


# ## Hilbert Transform

# In[33]:


### Hilbert Transform Process ######
# create a differenced series ########

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - 2 * dataset[i - interval] + dataset[i - 2 * interval]
        diff.append(value)
        
    return np.array(diff)



import statistics 
from scipy.signal import argrelextrema
import itertools
def finding_maximum(signal, threshold,fs):
    # for local maxima
    max_signal = argrelextrema(signal, np.greater)
    max_signal_list = list(itertools.chain.from_iterable(max_signal))
    max_signal_numpy = np.array(max_signal_list)
    
    del_index = []
    for i in range(len(max_signal_numpy)):
        
        if signal[max_signal_numpy[i]]<threshold:
            
            del_index.append(max_signal_numpy[i])
            
    max_index = [ele for ele in max_signal_numpy if ele not in del_index]
    
    
    del_index_2 = []
    for i in range(1,len(max_index)-1):
        
        if (max_index[i] - max_index[i-1]) < (0.35/(1/fs)) and (signal[max_index[i]] < signal[max_index[i-1]]):
            
            del_index_2.append(max_index[i])
        if (max_index[i+1] - max_index[i]) < (0.35/(1/fs)) and (signal[max_index[i]] < signal[max_index[i+1]]):
            del_index_2.append(max_index[i])
            
    final_max = [ele for ele in max_index if ele not in del_index_2]
    
    return final_max

from scipy.signal import hilbert
def hilbert_transform(signal,fs):
    
    derivative = np.diff(signal, n=2)
    
    analytic_signal = hilbert(derivative)
    
    threshold = 1.35 * statistics.mean(abs(analytic_signal))
    
    max_indeces = finding_maximum(abs(analytic_signal), threshold, fs)
    
    total_index = []
    for i in range(len(max_indeces)):

        area = np.array(signal[max_indeces[i] - int(0.25*fs): max_indeces[i] + int(0.25*fs)])

        max_signal = argrelextrema(area, np.greater)
        max_signal_list = list(itertools.chain.from_iterable(max_signal))
        max_signal_numpy = np.array(max_signal_list)

        if len(max_signal_numpy)>0:
            peak_list = np.array(area[max_signal_numpy])
            peak_position = np.where(area==max(peak_list))

            total_index.append(peak_position[0]+max_indeces[i] - int(0.25*fs))

    total_index = np.asarray(total_index)
    
    return total_index


# ## Adaptive Threshold

# In[28]:


def adaptive_threshold(signal):
    
    df = pd.DataFrame(signal)
    avr_forw = df.rolling(window=3).mean()
    avr_forw[0:2] = df[0:2]
    
    avr_back = avr_forw.rolling(window=3).mean().shift(-3)
    avr_back[len(avr_back)-2:len(avr_back)] = avr_forw[len(avr_back)-2:len(avr_back)]
    
    avr_back = np.asarray(signal)
    
    possible_peak = []
    possible_peak_index = []
    possible_valley = []
    possible_valley_index = []
    
    for j in range(1, len(avr_back)-1):
        
        if (avr_back[j]>avr_back[j-1] and avr_back[j]>avr_back[j+1]):
            
            possible_peak.append(avr_back[j])
            possible_peak_index.append(j)
            
        if (avr_back[j]<avr_back[j-1] and avr_back[j]<avr_back[j+1]):
            
            possible_valley.append(avr_back[j])
            possible_valley_index.append(j)
            
    if possible_peak_index[0]<possible_valley_index[0]:
        
        del possible_peak_index[0]
        del possible_peak[0]
            
    copy_peak_index = possible_peak_index.copy()
    copy_peak = possible_peak.copy()
    copy_valley = possible_valley.copy()
    copy_valley_index = possible_valley_index.copy()
    
    total_peak_indeces = [] 
    flag = 0
    while (flag == False):
        
        VPD = []
        
        if len(copy_peak) <= len(copy_valley):
            length = len(copy_peak)
        elif len(copy_peak) > len(copy_valley):
            length = len(copy_valley)
        
        for j in range(length):
            
            diff = copy_peak[j] - copy_valley[j]
            VPD.append(diff)


        del_index_peak = []
        del_peak = []
        del_valley = []
        
        for p in range(1, len(VPD)-1):
            
            if VPD[p] < 0.25 * ((VPD[p-1] + VPD[p] + VPD[p+1])/3):
                del_index_peak.append(copy_peak_index[p])
                del_peak.append(copy_peak[p])
                del_valley.append(copy_valley[p])
        
        if (len(del_index_peak) == 0 or len(del_peak) == 0 or len(del_valley) == 0):
            flag = 1
            
        copy_peak_index = [ele for ele in copy_peak_index if ele not in del_index_peak]
        copy_peak = [ele for ele in copy_peak if ele not in del_peak]
        del_valley = [ele for ele in del_valley if ele not in del_valley]
        
    return copy_peak_index


# In[29]:


def peak_pred(ypred, test_X, test_y):
    
    test = normalize(ypred)

    signal = np.concatenate(test_X)
    
    j = 0
    indeces = []
    binary_indeces_CNN = np.zeros(len(test))
    while (j < len(test)-3):
        
        if test[j]>= 0.55:
            
            if j< len(test)-15:
                
                period = test[j:j+15]
                index = np.asarray(np.where(period==np.max(period)))
                if len(index[0])>1:
                    length = len(index[0])
                    index = index.tolist()
                    index[0][1:length] = []
                    indeces.append(int(index[0][0]+j))
                    np.put(binary_indeces_CNN, int(index[0][0]+j), [1] )
                    
                else:
                    indeces.append(int(index[0]+j))
                    np.put(binary_indeces_CNN, int(index[0]+j), [1] )

                j = j+15
            else:
                period = test[j:j+7]
                index = np.asarray(np.where(period==np.max(period)))

                indeces.append(int(index[0]+j))
                np.put(binary_indeces_CNN, int(index[0]+j), [1] )

                j = j+7
            
        else:
            j +=1


    e = 0
    while (e<len(indeces)-2):
        if (indeces[e+1]-indeces[e]<35):
            if (indeces[e+2]-indeces[e+1]<35):
                del (indeces[e+1])
            e += 1

        else:
            e += 1
            
    for k in range(len(indeces)):
        if (indeces[k]-5>0) and (indeces[k]+5<len(signal)):
            period = signal[indeces[k]-5:indeces[k]+5]
            period = period.tolist()
            peak_ind = indeces[k]-5 + period.index(max(period))
            indeces[k] = peak_ind
            

    binary_indeces_CNN = np.zeros(len(test))
    np.put(binary_indeces_CNN, indeces, [1] )
    
    return indeces


# ## Building the Confusion Matrix

# In[30]:


def Confusion_Matrix(ypred, test_X, test_y):
    import heartpy as hp
    import neurokit2 as nk


    total_index = []
    total_binary_CNN = []
    total_binary_indeces_NK = []
    total_binary_indeces_HP = []
    Heartpy_peak = []
    discard = []
    discard_N = []
    test = normalize(ypred)

    signal = np.concatenate(test_X)
    
    j = 0
    indeces = []
    binary_indeces_CNN = np.zeros(len(test))
    binary_indeces_HP = np.zeros(len(test))
    binary_indeces_NK = np.zeros(len(test))
    while (j < len(test)-3):
        
        if test[j]>= 0.60:
            
            if j< len(test)-15:
                
                period = test[j:j+15]
                index = np.asarray(np.where(period==np.max(period)))
                if len(index[0])>1:
                    length = len(index[0])
                    index = index.tolist()
                    index[0][1:length] = []
                    indeces.append(int(index[0][0]+j))
                    np.put(binary_indeces_CNN, int(index[0][0]+j), [1] )
                    
                else:
                    indeces.append(int(index[0]+j))
                    np.put(binary_indeces_CNN, int(index[0]+j), [1] )

                j = j+15
            else:
                period = test[j:j+7]
                index = np.asarray(np.where(period==np.max(period)))

                indeces.append(int(index[0]+j))
                np.put(binary_indeces_CNN, int(index[0]+j), [1] )

                j = j+7
            
        else:
            j +=1


    e = 0
    while (e<len(indeces)-2):
        if (indeces[e+1]-indeces[e]<30):
            if (indeces[e+2]-indeces[e+1]<30):
                del (indeces[e+1])
            e += 1

        else:
            e += 1

    binary_indeces_CNN = np.zeros(len(test))
    np.put(binary_indeces_CNN, indeces, [1] )


    ############ True Peaks ################

    True_peak = test_y
    
    ##### Heartpy analysis #########
    ################################
    try:
        wd, m = hp.process(signal, sample_rate=100)
        heartpy_peak = wd['binary_peaklist']*wd['peaklist']
        H_peak = np.array([i for i in heartpy_peak if i != 0])
        np.put(binary_indeces_HP, H_peak, [1] )
     
        
    except:
        discard.append(1)
    if len(discard)==0:
        TP_HP, FP_HP, FN_HP = metrics(True_peak, H_peak, binary_indeces_HP)

    else:
        TP_HP = [0]
        FP_HP = [0]
        FN_HP = [0]
    
    ############## Neurokit ##############
    ######################################
    try:
        info = nk.ppg_findpeaks(signal, sampling_rate=100)
        Neurokit_peak = info['PPG_Peaks']
        np.put(binary_indeces_NK, Neurokit_peak, [1] )
 
    except:
        discard_N.append(1)

    if len(discard_N)==0:
        TP_NK, FP_NK, FN_NK = metrics(True_peak, Neurokit_peak, binary_indeces_NK)

    else:
        TP_NK = [0]
        FP_NK = [0]
        FN_NK = [0]
   

    
    ############# Finding the TP and FP and FN for CNN  ############
    #################################################################
    TP_CNN, FP_CNN, FN_CNN = metrics(True_peak, indeces, binary_indeces_CNN)

    
    ################ Hilbert Transform ##############################
    ##################################################################
    H_peak = hilbert_transform(signal, 100)
    binary_indeces_HT = np.zeros(len(test))
    if len(H_peak)==0:
        binary_indeces_HT = np.zeros(len(test))
    else:
        np.put(binary_indeces_HT, H_peak, [1] )
    if len(H_peak)==0:
        TP_HT = [0]
        FP_HT = [0]
        FN_HT = [0]
    else:
        TP_HT, FP_HT, FN_HT = metrics(True_peak, H_peak, binary_indeces_HT)
        


     ######################## Adaptive Threshold ############################
    ###################################################################
    A_peak = adaptive_threshold(test_X)
    binary_indeces_AT = np.zeros(len(test))
    np.put(binary_indeces_AT, A_peak, [1] )
    if len(A_peak)==0:
        TP_AT = [0]
        FP_AT = [0]
        FN_AT = [0]
    else:
        TP_AT, FP_AT, FN_AT = metrics(True_peak, A_peak, binary_indeces_AT)
    
    
    return TP_CNN, FP_CNN, FN_CNN, TP_NK, FP_NK, FN_NK, TP_HP, FP_HP, FN_HP, TP_HT, FP_HT, FN_HT,TP_AT, FP_AT, FN_AT
    

