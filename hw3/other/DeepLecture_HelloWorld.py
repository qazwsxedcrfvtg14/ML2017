import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
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
        if cnt==1000:
            break
        sp=line.split(',')
        y_train.append(float(sp[0]))
        x_train.append([float(v) for v in sp[1].split(' ')])
    f=open("test.csv","r")
    f.readline()
    x_test=[]
    y_test=[]
    while True:
        line=f.readline()
        if not line:
            break
        sp=line.split(',')
        y_test.append(float(sp[0]))
        x_test.append([float(v) for v in sp[1].split(' ')])
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

#(x_train,y_train),(x_test,y_test)=load_data()
(x_train,y_train),(x_test,y_test)=load_data()

"""
model = Sequential()
model.add(Convolution2D(32,33,3,input_shape=(48,48,1)))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(64,3,3))
model.add(MaxPooling2D(2,2))
model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(7))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=100,epochs=20)

score = model.evaluate(x_train,y_train,batch_size=10000)
print 'Train Acc:', score[1]
score = model.evaluate(x_test,y_test,batch_size=10000)
print 'Test Acc:', score[1]
"""
model2 = Sequential()
#model2.add(Convolution2D(25,3,3,input_shape=(48,48,1)))
model2.add(Convolution2D(32,33,3,input_shape=(48,48,1)))
model2.add(MaxPooling2D((2,2)))
#model2.add(Convolution2D(50,3,3))
model2.add(Convolution2D(64,3,3))
model2.add(MaxPooling2D((2,2)))
model2.add(Flatten())
model2.add(Dense(100))
model2.add(Activation('relu'))
model2.add(Dense(10))
model2.add(Activation('softmax'))
model2.summary()

model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

x_train=x_train.reshape(x_train.shape[0],48,48,1)
x_test=x_test.reshape(x_test.shape[0],48,48,1)

model2.fit(x_train,y_train,batch_size=100,epochs=20)

score = model2.evaluate(x_train,y_train,batch_size=10000)
print('Train Acc:', score[1])
score = model2.evaluate(x_test,y_test,batch_size=10000)
print('Test Acc:', score[1])


