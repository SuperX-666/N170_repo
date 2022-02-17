# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:53:01 2020

@author: pangh
"""

# In[7]:Epoch

# events
events, event_id = mne.events_from_annotations(data)
events_pick = mne.pick_events(events, exclude=[1,6,7,8])
# Mapping Event IDs to trial descriptors
event_dict = {'face/up': 2, 'chair/up': 3, 'face/down': 4,'chair/down': 5}

# In[7]:Epoch

# Creating Epoched data
epochs = mne.Epochs(data_mastoid_ref, events_pick, event_id=event_dict, 
                    tmin=-0.2, tmax=0.8, preload=True)#baseline correction was automatically applied 

# In[8]:

# Check & drop epoch manually
fig = epochs.plot(picks=['eeg'], scalings=dict(eeg=100e-6), n_epochs=5, n_channels=63, block=False)
plt.show()

# In[4]:Resampling
data_downsampled = epochs.resample(sfreq=250)

# In[9]:ICA

from mne.preprocessing import ICA

ica = ICA(n_components=61, max_pca_components=61,random_state=None, method='infomax')
ica.fit(epochs)

# 或直接
# ica = mne.preprocessing.ICA（）

# In[9]:ICA
# epochs.load_data()
ica.plot_sources(epochs, stop=3)
plt.show()


# In[9]:ICA

ica.plot_components(inst=epochs)
plt.show()


# In[10]: ICA-remove EOG 

# Selecting ICA components manually
ica.exclude = [2]  # indices chosen based on various plots above

# ica.apply() changes the Raw object in-place, so let's make a copy first:
epoch_data = epochs.copy()
ica.apply(epoch_data)


# In[ ]:
# ica.plot_properties(epochs, picks=ica.exclude)


# In[11]:Manually reject bad epochs


fig = epoch_data.plot(picks=['eeg'], scalings=dict(eeg=100e-6), n_epochs=2, n_channels=63)
plt.show()

# In[12]ERP analysis
'''
EvokedArray(data, info, tmin=0.0, comment='', nave=1, kind='average', verbose=None)[source]
data: array of shape (n_channels, n_times)
'''


# In[65]:


# 保证两个条件使用等量的trial数（有问题，待调整）
epoch_data.equalize_event_counts(['face','chair'])

face_evoked = epoch_data['face'].average()#MNE-Python can select based on partial matching of /-separated epoch labels
chair_evoked = epoch_data['chair'].average()


# In[108]:保存 ERP 数据

face_evoked.save('results/face_eeg-ave.fif')  # save evoked data to disk
chair_evoked.save('results/chair_eeg-ave.fif')

# In[91]:差异波


diff_evoked = mne.combine_evoked([face_evoked, chair_evoked], weights=[1,-1])
diff_evoked.plot(picks=['P8'])
plt.show()

# OR
# diff_data = face_evoked.data-chair_evoked.data

# In[51]:Butterfly plot

fig = plt.figure()
ax2 = fig.add_subplot(121)
ax3 = fig.add_subplot(122)
face_evoked.plot(gfp=True, spatial_colors=True, axes=ax2) #default exclude='bads'
chair_evoked.plot(gfp=True, spatial_colors=True, axes=ax3)
plt.show()


# In[53]:


# Joint plot
face_evoked.plot_joint(times=0.17)
plt.show()


# Scalp maps

# In[42]:


# scalp topographies
# single time point
fig=face_evoked.plot_topomap(ch_type='eeg', times=0.17, average=0.01, colorbar=True)#outlines='skirt'to see the topography stretched beyond the head circle
fig.text(0.5, 0.05, 'average from 165-175 ms', ha='center')
plt.show()


# In[35]:


# scalp topographies
# multiple time points
times = np.arange(0.1, 0.3, 0.02)#间隔0.02s
face_evoked.plot_topomap(ch_type='eeg', times=times, colorbar=True,
                         ncols=5, nrows='auto')
plt.show()


# In[47]:


# Animating the topomap
times = np.arange(0.1, 0.5, 0.01)
fig, anim = face_evoked.animate_topomap(
    times=times, ch_type='eeg', frame_rate=2, time_unit='s', blit=False, 
    butterfly=True)


# ERP traces

# In[96]:


# Comparing Evoked objects
mne.viz.plot_compare_evokeds([face_evoked, chair_evoked], picks='P8', ylim=dict(eeg=[-20,20]))
plt.show()


# In[77]:


# Topographical subplots
mne.viz.plot_compare_evokeds([face_evoked, chair_evoked], picks='eeg', axes='topo')
# OR
# mne.viz.plot_evoked_topo([face_evoked, chair_evoked])


# Evoked arithmetic (e.g. differences)

# In[80]:


# create and plot difference ERP
joint_kwargs = dict(ts_args=dict(time_unit='s'),
                    topomap_args=dict(time_unit='s'))
mne.combine_evoked([face_evoked, chair_evoked], weights=[1, -1]).plot_joint(**joint_kwargs)
plt.show()


# Peak detection

# In[137]:


# View evoked response
# times = 1e3 * epochs.times  # time in miliseconds

ch_max_name, latency, amplitude = face_evoked.get_peak(ch_type='eeg', tmin=0.15, tmax=0.2, mode='neg', return_amplitude=True)

face_evoked.plot(picks=ch_max_name)
plt.show()
