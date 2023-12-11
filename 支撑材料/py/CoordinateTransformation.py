#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import torch
import numpy as np

# In[2]:


label_path = 'potholes\labels'
image_path = 'potholes\images'


def xywhn2xyxy(x, w=416, h=416, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


folder = os.path.exists('potholes\\labelsNew')
if not folder:
    os.makedirs('potholes\\labelsNew')

folderlist = os.listdir(label_path)
for i in folderlist:
    label_path_new = os.path.join(label_path, i)
    with open(label_path_new, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        print(lb)
    read_label = label_path_new.replace(".txt", ".jpg")
    read_label_path = read_label.replace(label_path, image_path)
    print(read_label_path, "ddd")
    img = cv2.imread(str(read_label_path))
    h, w = img.shape[:2]
    lb[:, 1:] = xywhn2xyxy(lb[:, 1:], w, h, 0, 0)

    for _, x in enumerate(lb):
        class_label = int(x[0])

        with open('potholes\\labelsNew\\' + i, 'a') as fw:
            fw.write(str(int(x[0])) + ' ' + str(x[5]) + ' ' + str(x[1]) + ' ' + str(x[2]) + ' ' + str(x[3]) + ' ' + str(
                x[4]) + '\n')

# In[ ]:
