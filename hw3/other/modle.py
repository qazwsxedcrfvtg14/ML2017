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
    f=open("train.csv","r")
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
    f=open("test.csv","r")
    f.readline()
    x_test=[]
    y_test=[]
    cnt=0
    while True:
        line=f.readline()
        if not line:
            break
        cnt+=1
        #if cnt==2:
        #    break
        sp=line.split(',')
        y_test.append([0 for v in range(7)])
        #y_test.append([0 for v in range(10)])
        #y_test[-1][int(sp[0])]=1
        #y_test.append(float(sp[0]))
        x_test.append([float(v)/256 for v in sp[1].split(' ')])

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

(x_train,y_train),(x_test,y_test)=load_data()
x_train = x_train.reshape(x_train.shape[0],48,48,1)
x_test = x_test.reshape(x_test.shape[0],48,48,1)
#print(x_train)
#quit()
"""
model = Sequential()
model.add(Dense(input_dim=48*48,units=689,activation='relu'))
model.add(Dense(units=689,activation='relu'))
model.add(Dense(units=689,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=100,epochs=20)

score = model.evaluate(x_train,y_train)
print('\nTrain Acc:', score[1])
score = model.evaluate(x_test,y_test)
print('\nTest Acc:', score[1])
"""

rate=0.2

dropout_rate = 0.35
lmbda = 1e-5
"""
x = Input(shape = (48, 48, 1))
x = BatchNormalization()(x)
c1 = Conv2D(32, 1, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = l2(lmbda), activation='relu')(x)
c1 = MaxPooling2D((2, 2))(c1)
c1 = Dropout(dropout_rate)(c1)
c2 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = l2(lmbda), activation='relu')(x)
c2 = MaxPooling2D((2, 2))(c2)
c2 = Dropout(dropout_rate)(c2)
c3 = Conv2D(32, 5, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = l2(lmbda), activation='relu')(x)
c3 = MaxPooling2D((2, 2))(c3)
c3 = Dropout(dropout_rate)(c3)
#c4 = Conv2D(64, 7)(x)
#c4 = MaxPooling2D((2, 2))(c4)
#c4 = Dropout(dropout_rate)(c4)
#c5 = Conv2D(64, 9)(x)
#c5 = MaxPooling2D((2, 2))(c5)
#c5 = Dropout(dropout_rate)(c5)
c = concatenate([c1, c2, c3])
c = BatchNormalization()(c)

c1 = Conv2D(64, 1, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = l2(lmbda), activation='relu')(c)
c1 = MaxPooling2D((2, 2))(c1)
c1 = Dropout(dropout_rate)(c1)
c2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = l2(lmbda), activation='relu')(c)
c2 = MaxPooling2D((2, 2))(c2)
c2 = Dropout(dropout_rate)(c2)
c3 = Conv2D(64, 5, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = l2(lmbda), activation='relu')(c)
c3 = MaxPooling2D((2, 2))(c3)
c3 = Dropout(dropout_rate)(c3)
#c4 = Conv2D(128, 4)(c)
#c4 = MaxPooling2D((2, 2))(c4)
#c4 = Dropout(dropout_rate)(c4)
#c5 = Conv2D(128, 5)(c)
#c5 = MaxPooling2D((2, 2))(c5)
#c5 = Dropout(dropout_rate)(c5)
c = concatenate([c1, c2, c3])
c = BatchNormalization()(c)

c1 = Conv2D(64, 1, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = l2(lmbda), activation='relu')(c)
c1 = MaxPooling2D((2, 2))(c1)
c1 = Dropout(dropout_rate)(c1)
c2 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = l2(lmbda), activation='relu')(c)
c2 = MaxPooling2D((2, 2))(c2)
c2 = Dropout(dropout_rate)(c2)
c3 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = l2(lmbda), activation='relu')(c)
c3 = MaxPooling2D((2, 2))(c3)
c3 = Dropout(dropout_rate)(c3)
c = concatenate([c1, c2, c3])
c = BatchNormalization()(c)

c1 = Conv2D(64, 1, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = l2(lmbda), activation='relu')(c)
c1 = MaxPooling2D((2, 2))(c1)
c1 = Dropout(dropout_rate)(c1)
c2 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = l2(lmbda), activation='relu')(c)
c2 = MaxPooling2D((2, 2))(c2)
c2 = Dropout(dropout_rate)(c2)
c3 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'glorot_normal', kernel_regularizer = l2(lmbda), activation='relu')(c)
c3 = MaxPooling2D((2, 2))(c3)
c3 = Dropout(dropout_rate)(c3)
c = concatenate([c1, c2, c3])
c = BatchNormalization()(c)

#c=c1
c = Flatten()(c)
y = Dense(64, activation = 'relu')(c)
y = Dropout(dropout_rate)(y)
y = Dense(1024, activation = 'relu')(y)
y = Dropout(dropout_rate)(y)
y = Dense(7, activation = 'softmax')(y)
model2 = Model(inputs = [x], outputs = [y])
model2.summary()
"""
"""
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
"""
"""
model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
"""
"""
model2.fit_generator(datagen.flow(x_train, y_train, batch_size=100),
                    steps_per_epoch=len(x_train), epochs=100,validation_split=0.3)
"""
"""
for e in range(100):
    print('Epoch', e)
    batches = 0
    batch_size=100
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):
        model2.fit(x_batch, y_batch,validation_split=0.3)
        batches += 1
        if batches >= len(x_train) / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
"""
"""
model2.fit(x_train,y_train,batch_size=100,epochs=100,validation_split=0.3)
"""

"""
model2 = Sequential()
model2.add(Conv2D(25,(3,3),input_shape=(48,48,1)))
model2.add(BatchNormalization())
#model2.add(MaxPooling2D((2,2)))
#model2.add(AveragePooling2D((2,2)))

model2.add(Dropout(0.3))

model2.add(Conv2D(50,(3,3)))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2,2)))
#model2.add(AveragePooling2D((2,2)))
model2.add(Dropout(0.3))

model2.add(Conv2D(100,(3,3)))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2,2)))
#model2.add(AveragePooling2D((2,2)))
model2.add(Dropout(0.3))

model2.add(Conv2D(125,(3,3)))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2,2)))
#model2.add(AveragePooling2D((2,2)))
model2.add(Dropout(0.3))

model2.add(Conv2D(250,(3,3)))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D((2,2)))
#model2.add(AveragePooling2D((2,2)))
model2.add(Dropout(0.4))
model2.add(Flatten())

model2.add(Dense(units=1000,activation='relu'))
model2.add(Dense(units=7,activation='softmax'))
"""




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

#x_train = x_train.reshape(x_train.shape[0],48,48,1)
#x_test = x_test.reshape(x_test.shape[0],48,48,1)
#model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
#model2.fit(x_train,y_train,batch_size=100,epochs=100,validation_split=0.3)


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
#score = model2.evaluate(x_test,y_test)
#print('\nTest Acc:', score[1])
y_test=model2.predict(x_test)
result=[]
for i in y_test:
    ma=0
    res=0
    for j in range(len(i)):
        if i[j]>ma:
            #result.append(j)
            ma=i[j]
            res=j
    result.append(res)
print(result)
f=codecs.open('out.csv', 'w', 'Big5')
f.write("id,label\n")
for i in range(len(result)):
    f.write(str(i)+","+str(result[i])+"\n")
f.close()

model2.save('my_model')
