#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

dataDetect = pd.read_csv('detectResults.txt', sep=' ', header=None)
dataSegment = pd.read_csv('segmentResults.txt', sep=' ', header=None)

# In[2]:


dataDetect = dataDetect.drop([0, 1, 3, 5, 6], axis=1)
dataDetect.columns = ['fnames', 'Detect-Npotholes']
dataDetect['fnames'] = dataDetect['fnames'].str.split('\\').str[-1]
dataDetect['Detect-Npotholes'] = dataDetect['Detect-Npotholes'].str.replace('(', '')
dataDetect['fnames'] = dataDetect['fnames'].str.replace(':', '')
dataDetect['Detect-Npotholes'] = dataDetect['Detect-Npotholes'].str.replace('no', '0')
dataDetect['Detect-Npotholes'] = dataDetect['Detect-Npotholes'].astype('int')
dataDetect

# In[3]:


dataSegment = dataSegment.drop([0, 1, 3, 5, 6], axis=1)
dataSegment.columns = ['fnames', 'Segment-Npotholes']
dataSegment['fnames'] = dataSegment['fnames'].str.split('\\').str[-1]
dataSegment['Segment-Npotholes'] = dataSegment['Segment-Npotholes'].str.replace('(', '')
dataSegment['fnames'] = dataSegment['fnames'].str.replace(':', '')
dataSegment['Segment-Npotholes'] = dataSegment['Segment-Npotholes'].str.replace('no', '0')
dataSegment['Segment-Npotholes'] = dataSegment['Segment-Npotholes'].astype('int')
dataSegment

# In[4]:


data = pd.merge(dataDetect, dataSegment, on='fnames', how='left')
data

# In[5]:


# DetectFalse为Detect-Npotholes为0的个数
# DetectTrue为Detect-Npotholes不为0的个数
# SegmentFalse为Segment-Npotholes为0的个数
# SegmentTrue为Segment-Npotholes不为0的个数
DetectFalse = 0
DetectTrue = 0
SegmentFalse = 0
SegmentTrue = 0
for i in range(len(data)):
    if data['Detect-Npotholes'][i] == 0:
        DetectFalse += 1
    else:
        DetectTrue += 1
    if data['Segment-Npotholes'][i] == 0:
        SegmentFalse += 1
    else:
        SegmentTrue += 1

# In[6]:


DetectTrue / (DetectFalse + DetectTrue)

# In[7]:


SegmentTrue / (SegmentFalse + SegmentTrue)
