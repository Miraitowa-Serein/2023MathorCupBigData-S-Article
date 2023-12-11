#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ultralytics import YOLO


# In[47]:


model = YOLO("potholesDetect.pt")
resultsOne = model.predict(source="potholes\\partOne", save=True, save_conf=True, save_txt=True,imgsz=416)


# In[48]:


model = YOLO("potholesDetect.pt")
resultsTwo = model.predict(source="potholes\\partTwo", save=True, save_conf=True, save_txt=True,imgsz=416)

