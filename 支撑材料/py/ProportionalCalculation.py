#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import pandas as pd

# In[2]:


labelsList = os.listdir('potholes\\labelsNew')
imagesList = os.listdir('potholes\\images')

# In[3]:


labelsDf = pd.DataFrame(labelsList, columns=['fnames'])
labelsDf

# In[4]:


Npotholes = pd.read_csv('potholes\\detectResults.txt', sep=' ', header=None)
Npotholes = Npotholes.drop([0, 1, 3, 5, 6], axis=1)
Npotholes.columns = ['fnames', 'Npotholes']
Npotholes['fnames'] = Npotholes['fnames'].str.split('\\').str[-1]
Npotholes['Npotholes'] = Npotholes['Npotholes'].str.replace('(', '')
Npotholes['fnames'] = Npotholes['fnames'].str.replace(':', '')
Npotholes

# In[5]:


Npotholes['Npotholes'] = Npotholes['Npotholes'].str.replace('no', '0')
Npotholes['Npotholes'] = Npotholes['Npotholes'].astype('int')
Npotholes

# In[6]:


data = pd.DataFrame(columns=['Type', '置信度', 'x1', 'y1', 'x2', 'y2', 'fnames'])
data

# In[7]:


for i in labelsList:
    with open('potholes\\labelsNew\\' + i, 'r') as f:
        content = f.readlines()
        lb = np.array([x.strip().split() for x in content], dtype=np.float32)  # labels
        lb = pd.DataFrame(lb, columns=['Type', '置信度', 'x1', 'y1', 'x2', 'y2'])
        lb['fnames'] = i

    data = data._append(lb, ignore_index=True)

data

# In[8]:


data = data.drop('Type', axis=1)
data = data[['fnames', '置信度', 'x1', 'y1', 'x2', 'y2']]
data['fnames'] = data['fnames'].str.replace('.txt', '.jpg')
data

# In[9]:


data = pd.merge(data, Npotholes, on='fnames', how='right')
data

# In[10]:


potholesImages = pd.DataFrame(imagesList, columns=['fnames'])
potholesImages

# In[11]:


for i in imagesList:
    img = cv2.imread('potholes\\images\\' + i)
    height, width, channels = img.shape
    potholesImages.loc[potholesImages['fnames'] == i, 'height'] = height
    potholesImages.loc[potholesImages['fnames'] == i, 'width'] = width

potholesImages

# In[12]:


data = pd.merge(data, potholesImages, on='fnames', how='left')
data

# In[13]:


data['S'] = data['height'] * data['width']
data['X'] = data['x2'] - data['x1']
data['Y'] = data['y2'] - data['y1']
data['Spotholes'] = data['X'] * data['Y']
data['Spotholes/S'] = data['Spotholes'] / data['S'] * 100
data

# In[14]:


data['Spotholes/S'] = data['Spotholes/S'].fillna(data['Spotholes/S'].mean())
data['Spotholes/S'] = data['Spotholes/S'].apply(np.ceil)
data['Spotholes/S'] = data['Spotholes/S'].astype('int')
data

# In[15]:


dataNew = data.groupby('fnames').apply(lambda x: x.loc[x['Spotholes/S'].idxmax()])
dataNew

# In[16]:


dataNew = dataNew.reset_index(drop=True)
dataNew

# In[17]:


dataR = pd.read_csv('未知数据集预测效果.csv')
dataR = dataR.drop(['imgType', 'label', 'isTrue'], axis=1)
dataR

# In[18]:


Results = pd.merge(dataR, dataNew, on='fnames', how='left')
Results

# In[19]:


# 修改guy7iodk.jpg行，height=1587，width=1200，S=1587*1200=1904400，Spotholes=954792，Spotholes/S=50
Results.loc[Results['fnames'] == 'guy7iodk.jpg', 'height'] = 1587
Results.loc[Results['fnames'] == 'guy7iodk.jpg', 'width'] = 1200
Results.loc[Results['fnames'] == 'guy7iodk.jpg', 'S'] = 1904400
Results.loc[Results['fnames'] == 'guy7iodk.jpg', 'Spotholes'] = 954792
Results.loc[Results['fnames'] == 'guy7iodk.jpg', 'Spotholes/S'] = 50
Results

# In[20]:


Results.loc[Results['imgClass'] == 'normal', 'Spotholes/S'] = 0
Results

# In[21]:


Results['Spotholes/S'] = Results['Spotholes/S'].astype('int')
Results

# In[22]:


Results.to_csv('SpotholesS.csv', index=False)
