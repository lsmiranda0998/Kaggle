from keras.layers import *
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from scipy.misc import imsave
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
import os

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(176, 208, 3), padding='VALID'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='VALID'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='VALID'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Block 2
model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='VALID'))
model.add(AveragePooling2D(pool_size=(19, 19)))

# set of FC => RELU layers
model.add(Flatten())

#getting the summary of the model (architecture)
model.summary()
#
##iterates between all files from the directory
##for f in files:
#for folder, subdirs, files in os.walk(path):
#    for name in files:
#        #filter just excel files that is Breakdowns
#        if 'Breakdown' in name and '.xlsx' in name: #and 'Wrap' in name and 'xlsx' in name:

path = '/ALL-IDB2/'
name = 'Im001_1.tif'
images = []
names = []
labels = []
for folder, subdirs, files in os.walk(path):
    for name in files:
        if '.tif' in name:
            labels.append(name[6]==1?'Leucemia':'Control')
            img_path = folder+'/'+name
            img = image.load_img(img_path, target_size=(176, 208))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            images.append(img_data)

images = np.vstack(images)

vgg_feature = model.predict(images)
#print the shape of the output (so from your architecture is clear will be (1, 128))
#print shape
print(vgg_feature.shape)

#print the numpy array output flatten layer
print(vgg_feature.shape)

df = pd.DataFrame.from_records(vgg_feature)

#df[df['label'] == 'Alzheimer']

df['label'] = labels

sufix = folder.split('/')

df.to_csv(path+'features_'+sufix[len(sufix)-2]+'.csv',sep=';',index=False)