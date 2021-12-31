import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Flatten, Dense
import matplotlib.pyplot as plt
import cv2
from keras.applications.vgg19 import VGG19

trainDir='C:/Users/aashw/facemasks/Face Mask Dataset/Train'
testDir='C:/Users/aashw/facemasks/Face Mask Dataset/Test'
validDir='C:/Users/aashw/facemasks/Face Mask Dataset/Validation'
targetSize=(128,128)

train_gen=ImageDataGenerator(rescale=1.0/255,
                             horizontal_flip=(True),
                             shear_range=0.2,
                             zoom_range=0.2)
valid_gen=ImageDataGenerator(rescale=1.0/255,
                             horizontal_flip=(True),
                             shear_range=0.2,
                             zoom_range=0.2)
test_gen=ImageDataGenerator(rescale=1.0/255)

ds_train=train_gen.flow_from_directory(directory=trainDir,
                                       target_size=targetSize,
                                       class_mode='categorical',
                                       batch_size=32)
ds_valid=train_gen.flow_from_directory(directory=validDir,
                                       target_size=targetSize,
                                       class_mode='categorical',
                                       batch_size=32)
ds_test=train_gen.flow_from_directory(directory=testDir,
                                       target_size=targetSize,
                                       class_mode='categorical',
                                       batch_size=32)


vgg19=VGG19(weights='imagenet',include_top=False,input_shape=(128,128,3))
for layer in vgg19.layers:
    layer.trainable=False


model=Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(2,activation= 'sigmoid'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
history=model.fit(x=ds_train,
batch_size=32,
epochs=10,
verbose=2,
validation_data=ds_valid,
shuffle=True,
validation_batch_size=32,
)
model.evaluate(x=ds_test)
model.save('facemasknet.h5')


facemodel=cv2.CascadeClassifier('C:/Users/aashw/Downloads/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
img=cv2.imread('C:/Users/aashw/Face Mask Dataset/Train/WithMask/1484.png')
img=cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
faces=facemodel.detectMultiScale(img,scaleFactor=1.1)
outimg=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
for(x,y,w,h) in faces:
    cv2.rectangle(outimg,(x,y),(x+w,y+h),(0,0,255),1)
plt.figure(figsize=(12,12))
outimg
plt.imshow(outimg)

sampleImage=cv2.imread('C:/Users/aashw/Face Mask Dataset/Train/WithMask/1447.png')
sampleImage=cv2.resize(sampleImage,(128,128))
plt.imshow(sampleImage)
sampleImage=np.reshape(sampleImage, [1,128,128,3])
sampleImage=sampleImage/225.0
t=model.predict(sampleImage)

model=tf.keras.models.load_model('C:/Users/aashw/Desktop/facemasknet.h5')
mask_label = {0:'MASK',1:'NO MASK'}
dist_label = {0:(0,255,0),1:(255,0,0)}
def predict(dir):
    sampleImage=cv2.imread(dir)
    if len(faces)>=1:
        
        for i in range(len(faces)):
            (x,y,w,h) = faces[i]
            crop = cv2.resize(sampleImage,(128,128))
            crop = np.reshape(crop,[1,128,128,3])/255.0
            mask_result = model.predict(crop)
            t1=mask_result[0][0]
            if(t1>0.5):
                label=0
            else:
                label=1
            cv2.rectangle(sampleImage,(x,y),(x+w,y+h),dist_label[label],1)
        plt.figure(figsize=(15,15))
        plt.imshow(sampleImage)
                
    else:
        print("No. of faces detected ")
predict('C:/Users/aashw/Face Mask Dataset/Train/WithoutMask/138.png')
predict('C:/Users/aashw/Face Mask Dataset/Train/WithMask/1114.png')
