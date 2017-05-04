import numpy as np
import os
import sys
import csv
import codecs
import numpy
import scipy
import random
import math

os.environ["THEANO_FLAGS"] = "device=gpu0"
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.regularizers import l2
from keras.layers import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
#categorical_crossentropy

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))
numpy.set_printoptions(threshold=numpy.nan)
def load_data():
    f=open(sys.argv[1],"r")
    #f=open("train.csv","r")
    f.readline()
    x_train=[]
    y_train=[]
    cnt=0
    while True:
        line=f.readline()
        if not line:
            break
        cnt+=1
        #if cnt==3:
        #    break
        sp=line.split(',')
        y_train.append([0 for v in range(7)])
        y_train[-1][int(sp[0])]=1
        y_train.append([0 for v in range(7)])
        y_train[-1][int(sp[0])]=1
        #x_train.append([float(v)/256 for v in sp[1].split(' ')])
        s=[float(v)/256 for v in sp[1].split(' ')]
        x_train.append([])
        for i in range(48):
            for j in range(48):
                x_train[-1].append(s[i*48+j])
        #x_train[-1] = numpy.reshape(x_train[-1], (48, 48))
        #y_train.append([0 for v in range(7)])
        #y_train[-1][int(sp[0])]=1
        #print(scipy.ndimage.interpolation.rotate(numpy.array(x_train[-1]),3.1415926))
        #x_train.append(scipy.ndimage.interpolation.rotate(numpy.array(x_train[-1]),10))
        #y_train.append([0 for v in range(7)])
        #y_train[-1][int(sp[0])]=1
        #x_train.append(scipy.ndimage.interpolation.rotate(numpy.array(x_train[-1]),-10))
        x_train.append([])
        for i in range(48):
            for j in range(47,-1,-1):
                x_train[-1].append(s[i*48+j])
        #x_train[-1] = numpy.reshape(x_train[-1], (48, 48))
        #y_train.append([0 for v in range(7)])
        #y_train[-1][int(sp[0])]=1
        #x_train.append(scipy.ndimage.interpolation.rotate(numpy.array(x_train[-1]),10))
        #y_train.append([0 for v in range(7)])
        #y_train[-1][int(sp[0])]=1
        #x_train.append(scipy.ndimage.interpolation.rotate(numpy.array(x_train[-1]),-10))
        """
        ss=[float(v)/256 for v in sp[1].split(' ')]
        for i in range(47):
            for j in range(48):
                ss[i*48+j]+=s[(i+1)*48+j]
                ss[(i+1)*48+j]+=s[i*48+j]
        for i in range(48):
            for j in range(47):
                ss[i*48+j]+=s[i*48+j+1]
                ss[i*48+j+1]+=s[i*48+j]
            
        y_train.append([0 for v in range(10)])
        y_train[-1][int(sp[0])]=1
        y_train.append([0 for v in range(10)])
        y_train[-1][int(sp[0])]=1

        x_train.append([])
        for i in range(48):
            for j in range(48):
                x_train[-1].append(ss[i*48+j]/5)
        x_train.append([])
        for i in range(48):
            for j in range(47,-1,-1):
                x_train[-1].append(ss[i*48+j]/5)
        """
        #x_train.append([float(v) for v in sp[1].split(' ')])
    #f=open("test.csv","r")
    #f.readline()
    x_test=[]
    y_test=[]
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

(x_train,y_train),(x_test,y_test)=load_data()
x_train = x_train.reshape(x_train.shape[0],48,48,1)
rate=0.2

model2 = Sequential()
model2.add(BatchNormalization(input_shape=(48,48,1)))
model2.add(Conv2D(32,3))
model2.add(Activation('relu'))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.2))

model2.add(BatchNormalization())
model2.add(Conv2D(64,3))
model2.add(Activation('relu'))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.3))

model2.add(BatchNormalization())
model2.add(Conv2D(128,3))
model2.add(Activation('relu'))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.3))

model2.add(BatchNormalization())
model2.add(Conv2D(256,3))
model2.add(Activation('relu'))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(0.4))

model2.add(Flatten())

model2.add(Dense(units=2000,activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(units=7,activation='softmax'))


model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
x_val=x_train[round(len(x_train)*0.8):]
y_val=y_train[round(len(y_train)*0.8):]
x_train=x_train[:round(len(x_train)*0.8)]
y_train=y_train[:round(len(y_train)*0.8)]

datagen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #horizontal_flip=True
    )

datagen.fit(x_train)
model2.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                    steps_per_epoch=len(x_train)/16, epochs=300,validation_data=(x_val,y_val))



score = model2.evaluate(x_train,y_train)
print('\nTrain Acc:', score[1])
model2.save('my_model')
