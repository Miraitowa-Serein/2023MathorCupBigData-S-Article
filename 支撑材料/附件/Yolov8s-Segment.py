#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ultralytics import YOLO

# In[3]:


model = YOLO('yolov8s-seg.pt')
results = model.train(data="dataset/VOCdevkit/datasets/potholes.yaml", epochs=300, device='cpu')

# In[2]:


model = YOLO('potholesSegment.pt')

# In[3]:


resultOne = model('potholes\\part1', save=True)

# In[4]:


resultTwo = model('potholes\\part2', save=True)

# In[5]:


resultThree = model('potholes\\part3', save=True)

# In[3]:


resultFour = model('potholes\\part4', save=True)

# In[4]:


resultFive = model('potholes\\part5', save=True)

# In[3]:


resultSix = model('potholes\\part6', save=True)

# In[4]:


resultSeven = model('potholes\\part7', save=True)

# In[3]:


resultEight = model('potholes\\part8', save=True)
