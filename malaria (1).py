#!/usr/bin/env python
# coding: utf-8

# # Objectives
# 
# 1. Build a deep learning model to classify infected cells
# 2. Build a simple application using streamlit to classify a given image and highlight the hotspots in the image that influence the prediction
#     - The hotspots have to be overlayed on the image and displayed in the streamlit app
#     - Include a video demo of the app using a screen recorder
#     - Bonus: deploy the streamlit app on a cloud platform (eg: heroku)
# 3. Detail your approach and findings from the dataset in a 2-4 page technical report using a prefered format. Provide the necessary details to rationalize your assumptions and choice of methods.
# 4. Submit an archive containing the code and the report.
#     - If you used a kaggle kernel, publish it and share the kernel link

# # Evaluation criteria
# 
# The submissions will be evaluated on the following criteria
# 
# - The different approaches explored and the overall efficiency of the solution
# - The working of the application for classification and explanation
# - Effective communication of results through the technical paper
# - Performance of the final model

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


img_width= 128
img_height= 128


# In[3]:


datagen= ImageDataGenerator(rescale=1/255.0,validation_split=0.2)


# In[4]:


train_data_generator= datagen.flow_from_directory(directory='cell_images/cell_images',target_size=(img_width,img_height),
                                                  class_mode='binary',batch_size=16, subset='training')


# In[5]:


validation_data_generator= datagen.flow_from_directory(directory='cell_images/cell_images',target_size=(img_width,img_height),
                                                  class_mode='binary',batch_size=16, subset='validation')


# In[6]:


train_data_generator.labels


# # CNN

# In[7]:


from keras import Sequential
from keras.layers import Flatten,Dense,Dropout,Conv2D,MaxPool2D
from tensorflow.keras.optimizers import Adam


# In[8]:


model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))


# In[9]:


model.summary()


# In[10]:


model.compile(optimizer=Adam(),
             metrics=['Accuracy'],
             loss='binary_crossentropy')


# In[11]:


history = model.fit(train_data_generator,steps_per_epoch=len(train_data_generator),
         epochs=5,validation_data=validation_data_generator,
         validation_steps=len(validation_data_generator))


# In[ ]:




