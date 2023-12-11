#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model

# In[2]:


CNNModel = load_model('Models\\CNN.h5')
with open('Models\\CNN-SVM.pkl', 'rb') as f:
    model = pickle.load(f)

# In[3]:


imagePath = []
for dirname, _, filenames in os.walk('DATAPre\\'):
    print(dirname, len(filenames))
    for filename in filenames:
        path = os.path.join(dirname, filename)
        imagePath.append(path)

len(imagePath)

# In[4]:


IMG_SIZE = 128
X = []
y = []
imgName = []
for image in imagePath:
    try:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(np.array(img))
        imgName.append(image.split('\\')[1])
    except:
        pass

# In[5]:


X = np.array(X)
X_train_features = CNNModel.predict(X)

# In[6]:


y_pred = model.predict(X_train_features)
df = pd.DataFrame({'fnames': imgName, 'label': y_pred})
df['label'] = df['label'].astype(int)
df

# In[7]:


df.to_csv('test_result.csv', index=False)
