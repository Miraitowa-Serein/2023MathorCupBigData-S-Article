#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil

# 指定目录"DATA"
path = "DATA"

# 在"DATA"文件夹中创建"normal"和"potholes"文件夹
os.mkdir(os.path.join(path, "normal"))
os.mkdir(os.path.join(path, "potholes"))

# 读取"DATA"文件夹，若文件名中含有"normal"，则将其放置于"normal"文件夹中，否则放置于"potholes"文件夹中
files = os.listdir(path)
for file in files:
    if "normal" in file:
        shutil.move(os.path.join(path, file), os.path.join(path, "normal"))
    else:
        shutil.move(os.path.join(path, file), os.path.join(path, "potholes"))
