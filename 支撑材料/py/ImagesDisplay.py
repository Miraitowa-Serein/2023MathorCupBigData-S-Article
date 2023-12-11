#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2

img = cv2.imread("DATA\\potholes\\potholes1.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, [256, 256])

Gaussian = cv2.GaussianBlur(img, (3, 3), 1)
Bilateral = cv2.bilateralFilter(img, 9, 75, 75)
Rotate = cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 30, 1),
                        (img.shape[1], img.shape[0]))

cv2.imwrite("potholes1_Gaussian.jpg", Gaussian)
cv2.imwrite("potholes1_Bilateral.jpg", Bilateral)
cv2.imwrite("potholes1_Rotate.jpg", Rotate)
