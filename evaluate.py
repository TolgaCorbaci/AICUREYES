#!/usr/bin/env python
# coding: utf-8

# In[2]:


import itertools
import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import keras

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


# In[7]:


model_name1='mobilenet_v2_100_224_hack_heal'


# In[8]:


model = keras.models.load_model('/Users/viqor/Desktop/Jupiter/models/{}'.format(model_name1))


# In[4]:


model.summary()


# In[12]:


class_names = ['abnormal', 'normal']


# In[13]:


img_path = "/Users/viqor/Desktop/cropped2.jpg"


# In[17]:


from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os


def load_image(img_path, show=True):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":



    # load a single image
    new_image = load_image(img_path)

    # check prediction
    pred = model.predict(new_image)

    # check prediction
    prediction_scores = model.predict(new_image)
    predicted_index = np.argmax(prediction_scores)
    print("Predicted label: " + class_names[predicted_index])
    print(predicted_index)
        


# In[18]:


import sklearn
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# In[20]:


# Path to your folder testing data
testing_folder = "/Users/viqor/Desktop/Data/truefolders/test"
# Image size (set up the image size used for training)
img_size = 224
# Batch size (you should tune it based on your memory)
batch_size = 10

val_datagen = ImageDataGenerator(
    rescale=1. / 255)
validation_generator = val_datagen.flow_from_directory(
    testing_folder,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')


# In[21]:


# Number of steps corresponding to an epoch
steps = 57
predictions = model.predict(validation_generator, steps=steps)


# In[22]:


predictions[0][1]


# In[23]:


val_preds = np.argmax(predictions, axis=-1)
val_trues = validation_generator.classes
cm = metrics.confusion_matrix(val_trues, val_preds)


# In[24]:


precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(val_trues, val_preds, average= "binary")


# In[25]:


f1_score


# In[26]:


recall


# In[27]:


precisions


# In[28]:


from sklearn.metrics import classification_report, confusion_matrix


# In[29]:


print(model_name1)
Y_pred = predictions
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix \n')
print(confusion_matrix(validation_generator.classes, y_pred))
print('\nClassification Report\n')
target_names = ['anormal','normal']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

cm = confusion_matrix(validation_generator.classes, y_pred)
cm = pd.DataFrame(cm, range(2),range(2))
plt.figure(figsize = (10,10))

sns.heatmap(cm, annot=True, annot_kws={"size": 20},cmap="YlGnBu",fmt="d",yticklabels=["True_anormal","True_normal"],xticklabels=["Pred_anormal","Pred_normal"]) # font size
plt.show()

