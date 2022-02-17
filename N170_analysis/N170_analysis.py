#!/usr/bin/env python
# coding: utf-8

# In[0]:
import os
import numpy as np
import matplotlib.pyplot as plt
import mne

'''
结果存放设置
'''
def mkdir(path):
 # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
  
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
  
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
  
        print(path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False
  
# 定义要创建的目录
mkpath="results"
# 调用函数
mkdir(mkpath)

#%%
'''
Preprocessing
'''
#Loading data
data_folder = 'raw_data'
dataFile = 'n170_data.set'
data_file = os.path.join(data_folder,dataFile)
data = mne.io.read_raw_eeglab(data_file, eog = 'auto',preload=True)


# Channel location
data.drop_channels('EOG')
montage = mne.channels.read_custom_montage('standard-10-5-cap385.elp')
data.set_montage(montage)
# OR
# data.set_montage('standard_1020')

#%% check data manually
data.plot(duration=5, n_channels=63, scalings=dict(eeg=80e-6),block=True)
plt.show() # block模式（默认）下：绘图后会暂停执行，直到手动关闭当前窗口才继续执行后面的代码
plt.close()
#%% Filtering
data_bandpass = data.filter(l_freq=1, h_freq=35)
# data_notch = data_bandpass.notch_filter(freqs=50)

#%% check data manually
data_bandpass.plot(duration=5, n_channels=63, scalings=dict(eeg=80e-6),block=True)
# plt.show()

# In[6]:Reference
data_mastoid_ref = data_bandpass.set_eeg_reference(ref_channels=['M1', 'M2'], ch_type='eeg')

