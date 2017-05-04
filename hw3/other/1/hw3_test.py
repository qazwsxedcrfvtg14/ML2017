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
from keras.models import load_model
#categorical_crossentropy

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))
numpy.set_printoptions(threshold=numpy.nan)
def load_data():
    x_train=[]
    y_train=[]
    f=open(sys.argv[1],"r")
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
x_test = x_test.reshape(x_test.shape[0],48,48,1)

model2 = load_model("0.66787_modle")

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
f=codecs.open(sys.argv[2], 'w', 'Big5')
f.write("id,label\n")
for i in range(len(result)):
    f.write(str(i)+","+str(result[i])+"\n")
f.close()

