import numpy as np
import os
import sys
import csv
import codecs
import numpy
import random
import math
os.environ["THEANO_FLAGS"] = "device=gpu0"
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
#categorical_crossentropy


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
        #if cnt==10:
        #    break
        sp=line.split(',')
        y_train.append([0 for v in range(10)])
        y_train[-1][int(sp[0])]=1
        y_train.append([0 for v in range(10)])
        y_train[-1][int(sp[0])]=1
        #x_train.append([float(v)/256 for v in sp[1].split(' ')])
        s=[float(v)/256 for v in sp[1].split(' ')]
        x_train.append([])
        for i in range(48):
            for j in range(48):
                x_train[-1].append(s[i*48+j])
        x_train.append([])
        for i in range(48):
            for j in range(47,-1,-1):
                x_train[-1].append(s[i*48+j])
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
        y_test.append([0 for v in range(10)])
        #y_test.append([0 for v in range(10)])
        #y_test[-1][int(sp[0])]=1
        #y_test.append(float(sp[0]))
        x_test.append([float(v)/256 for v in sp[1].split(' ')])

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

(x_train,y_train),(x_test,y_test)=load_data()
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
model2 = Sequential()
model2.add(Conv2D(32,(3,3),input_shape=(48,48,1)))

model2.add(Dropout(rate))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(rate))
model2.add(Conv2D(64,(3,3)))
model2.add(Dropout(rate))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(rate))
model2.add(Conv2D(128,(3,3)))
model2.add(Dropout(rate))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(rate))
model2.add(Conv2D(256,(3,3)))
model2.add(Dropout(rate))
model2.add(MaxPooling2D((2,2)))
model2.add(Dropout(rate))

model2.add(Flatten())
model2.add(Dropout(rate))

model2.add(Dense(units=20000,activation='relu'))

model2.add(Dropout(rate))
model2.add(Dense(units=10,activation='softmax'))

x_train = x_train.reshape(x_train.shape[0],48,48,1)
x_test = x_test.reshape(x_test.shape[0],48,48,1)
model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model2.fit(x_train,y_train,batch_size=100,epochs=100,validation_split=0.3)


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