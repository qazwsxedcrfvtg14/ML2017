#!/usr/bin/python3
#[[0, 0, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 1, 1], [0, 1, 1, 0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 0, 1], [0, 1, 1, 0, 1, 1, 1, 1, 1]]
#[[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 0, 1], [0, 1, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0]]
#33.0202563382
import sys
import csv
import codecs
import numpy
import random
import math
#random.seed(str([7122,59491,66666,233333,9487]))
random.seed(59491)
usage=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
usage2=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
usage3=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
uslen=[0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0 ]
bestu=[1, 1, 1, 0, 6, 0, 4, 3, 1, 6, 2, 0, 7, 0, 0, 0, 4, 0]
# 36.6641645765 4.50574537742
bestu=[0, 1, 6, 0, 6, 0, 4, 3, 1, 6, 2, 0, 7, 0, 0, 0, 3, 0]
# 36.6316055946 4.50275650498
bestu=[0, 0, 1, 6, 5, 7, 6, 3, 1, 6, 5, 0, 6, 0, 0, 0, 1, 0]
# 34.672139131 4.52724285942
bestu=[0, 0, 3, 6, 5, 7, 6, 3, 1, 6, 5, 0, 6, 0, 0, 0, 1, 0]
# 34.9575354839 4.5328049763
#bestu=[0, 0, 0, 0, 5, 2, 6, 4, 1, 6, 1, 0, 6, 1, 0, 0, 1, 0]
# 35.1067963056 4.5296001407
bestu= [0, 0, 3, 7, 7, 7, 7, 3, 1, 7, 7, 0, 7, 0, 0, 0, 1, 0]
bestu2=[0, 0, 2, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0]
bestu3=[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

bestu= [0, 6, 3, 0, 2, 3, 0, 4, 1, 9, 0, 0, 1, 7, 0, 0, 0, 4]
bestu2=[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
bestu3=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

uslen=bestu[:]
uslen2=bestu2[:]
uslen3=bestu3[:]

#bestv=[[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 1, 1, 1], [1, 0, 1, 1, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 0, 0, 0]]
#bestv2=[[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
#bestv3=[[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 1, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0, 0, 0, 0]]
#[[1, 0, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 1, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0, 0, 1, 0]]
#[[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 0, 1, 1, 0, 1, 1, 0]]
"""
bestv=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 1, 1, 1], [1, 0, 1, 1, 0, 0, 1, 1, 1], 
    [1, 1, 1, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 1, 1, 1, 1], 
    [0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 1, 1, 0, 1, 0, 1, 1], [1, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 0, 1]]
bestv2=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1, 0, 0, 0]]
"""
"""
bestv=
[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0, 0], 
    [0, 0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]]
"""
bestv=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0]]
bestv=[
    [1, 1, 0, 0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 1, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 0, 0, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1, 1, 0, 1], [1, 0, 0, 1, 1, 1, 0, 1, 1]]
bestv=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
bestv=[
    [1, 1, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 1, 1, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0, 1], 
    [0, 1, 1, 0, 0, 0, 0, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]]

"""
bestv2=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0]]
"""
bestv2=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [0, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]

    
bestv2=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
bestv2=[
    [1, 1, 1, 1, 0, 1, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1], 
    [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 0, 0], 
    [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 0, 0], 
    [1, 0, 0, 0, 1, 1, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1, 0, 1, 1]]
bestv2=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
bestv3=[[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
bestv=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]
    
"""
bestv2=[
    [1, 0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0, 0, 1], [1, 0, 1, 0, 0, 0, 0, 0, 1], 
    [1, 1, 1, 0, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1, 0], 
    [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 0, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1], [1, 0, 1, 1, 1, 1, 1, 0, 1], 
    [1, 0, 0, 0, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 1, 0, 0, 1]]
"""
bestv=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0], 
    [1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0]]
bestv2=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]]

bestv3=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
"""
bestv=[
    [0, 1, 1, 0, 0, 0, 0, 0, 1], [0, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 0, 0, 1], 
    [1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 0, 0]]
bestv2=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0], 
    [1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1, 1], 
    [0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 0, 0]]
bestv3=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 1, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
bestv=[
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 1, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1, 0, 0, 1], 
    [1, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 1, 1, 1, 0, 1, 1], [1, 0, 1, 0, 0, 1, 1, 1, 1], [1, 1, 0, 1, 0, 1, 1, 1, 1], 
    [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0, 1, 1, 1]]
bestv2=[
    [1, 1, 1, 1, 1, 0, 0, 0, 1], [1, 1, 0, 1, 1, 0, 1, 0, 1], [1, 0, 1, 0, 0, 0, 1, 0, 0], [1, 1, 0, 1, 1, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1, 1, 0, 1], 
    [1, 1, 1, 0, 1, 0, 1, 1, 0], [1, 1, 1, 0, 0, 1, 1, 0, 1], [1, 1, 1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 0, 1, 0, 1, 1], 
    [1, 1, 1, 1, 1, 0, 1, 1, 1], [1, 0, 0, 0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 1, 1, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 1, 1, 1, 0], 
    [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 1, 1, 0, 0, 0]]
bestv3=[
    [1, 1, 0, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0, 1], 
    [1, 0, 0, 0, 0, 0, 1, 0, 0], [1, 1, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 1, 1, 0, 0], [1, 1, 0, 1, 0, 1, 0, 0, 1], [1, 0, 1, 1, 0, 1, 1, 1, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 1, 0], 
    [1, 0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
"""
#bestv3=[[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 0, 1, 0], [0, 1, 1, 0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
"""
[ -1.14720409e+00  -2.33785445e-01  -2.50726107e-01  -4.04978260e-01
   4.00443119e+00  -6.92124295e-02  -1.22434264e+00   8.88311949e-01
  -9.58809462e-02   8.77979333e-01  -6.46173701e-01  -3.32461120e-01
   5.35319606e-02  -5.68738466e-02   1.50860442e-01   1.37386727e-02
  -1.45121272e-01   2.17545157e-01  -1.86002895e-01  -6.22729816e-02
  -3.88471065e-02  -2.63296318e-03  -8.87437127e-02   3.53952615e-02
   1.35803190e-01   9.31926975e-02  -2.08919950e-02  -3.52033410e-02
  -8.15917447e-03   3.78598195e-02   2.28513465e-02  -2.19847025e-02
   5.10517794e-03   9.39233534e-01   3.65296055e-02  -5.52941180e-01
   4.67822569e-01  -2.31045024e-01   1.95647653e-01   6.15907070e-03
  -3.58594969e-02  -2.72703385e-02  -5.98473137e-02   6.65936188e-01
  -1.23767662e-01   1.71272427e-01  -1.12351115e-01   2.38701030e-01
  -5.48662081e-01  -9.67101833e-02  -2.09489927e-01  -6.74538646e-01
  -1.21986307e-01   9.04869364e-02  -1.51084159e-01  -1.59835402e-01
   3.25987488e-01  -8.57282750e-02   1.20411168e+00   3.92285238e-02
  -4.39732302e-01  -5.92665151e-01  -1.01295185e-01   1.84299540e-02
  -4.50236245e+00  -4.20818311e-01   4.05276690e+00  -3.51409254e+00
   1.13383357e+00   1.34717476e-02   1.05515937e-02  -8.14254038e-04
  -4.17394921e-02  -1.22774165e-02  -2.05085972e-02   5.74409076e-01
  -6.06631713e-02   2.04918159e-02   7.11382799e-03  -5.05461247e-02]
  """
vlen=bestv[:]
vlen2=bestv2[:]
vlen3=bestv3[:]

best=float('Inf')
result=open("result.txt","w")
#9 5.16081982185
#8 9.01713468749
#5 12.6097785164
#6 13.0510128224
#7 14.1102056211
#14 14.9979138787
#15 14.874133158
#12 14.0492186206
#uslen=bestu[:]
#uslen[p]=q
#id=5
id=9
#usage=[ 9, 8, 5, 6,12, 7,14,15,13, 1, 0]
#uslen=[ 8, 1, 5, 6, 5, 8, 4, 7, 8, 8, 8]

#usage=[ 9, 8, 5, 6,12, 7]
#uslen=[ 7, 1, 5, 6, 5, 7]

#usage=[0,5,6,7,8,9,12,16]
shift=0
tst=0
class Regression(object):
    def __init__ (self):
        #self.w_ = numpy.zeros(1 + 18*id)
        self.w_ = None #numpy.zeros(1 + id*len(usage))
        self.g_ = None
        self.t_ = 1e-20
        self.old_cost=float('Inf')
        self.eta=6e-2
    def rebuild(self):
        self.g_ = None
        self.t_ = 1e-20
        self.old_cost=float('Inf')
        self.eta=6e-2
    def fit(self,x,y,z):
        if self.w_ is None:
            self.w_=numpy.zeros(len(x[0])+1)
        #self.w_ = np.zeros(1 + 18*9) # make bias the zero one
        #cost=1
        #for i in range(0,self.n_iter):
        #print(x,self.w_[1:])
        #print(len(x))
        output = numpy.dot(x,self.w_[1:])+self.w_[0]
        errors = y- output 
        cost=numpy.sum(errors**2)/len(x)
        #if self.old_cost >= cost:
        self.old_cost=cost
        #self.eta=self.eta*1.0001
        #g=x.T.dot(errors).dot(z)
        #print("z=",z)
        #gg=x.T.dot(errors)
        g=x.T.dot(errors)
        #print(g-gg)
        t=errors.sum()
        if self.g_ is None:
            #self.g_=numpy.zeros(len(g))
            self.g_=numpy.full(len(g),1e-20)
        self.g_+=numpy.power(g,2)
        g/=numpy.sqrt(self.g_)
        self.t_+=t**2
        t/=self.t_**0.5
        self.w_[1:]+=self.eta*g
        self.w_[0]+=self.eta*t
        #print (np.sum(errors**2))
        #else:
        #    self.eta=self.eta*0.99
        #print (cost,self.eta)
        #print (cost)
        return self
    def activation(self,x):
        return numpy.dot(x,self.w_[1:])+self.w_[0]
#from scipy import linspace, polyval, polyfit, sqrt, stats, randn
#from pylab import plot, title, show , legend
table = [[], [], [], [], [], [], [], [],
        [], [], [], [], [], [], [], [], [], []]
ve = list(csv.reader(codecs.open(sys.argv[1], 'r', 'Big5')))
std = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
avg = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
cnt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(1, len(ve), 18):
    for j in range(18):
        for k in range(3,len(ve[i + j])):
            cnt[j]+=1
            if j == 14:
                val=math.sin(float(ve[i + 14][k]))*float(ve[i + 17][k])
            elif j==17:
                val=math.cos(float(ve[i + 14][k]))*float(ve[i + 17][k])
            elif j==15:
                val=math.sin(float(ve[i + 15][k]))*float(ve[i + 16][k])
            elif j==16:
                val=math.cos(float(ve[i + 15][k]))*float(ve[i + 16][k])
            else:
                try:
                    val=float(ve[i + j][k])
                except ValueError:
                    val=0.
            table[j].append(val)
            std[j]+=val**2
            avg[j]+=val
for j in range(18):
    avg[j]=avg[j]/cnt[j]
    std[j]=math.sqrt(std[j]/cnt[j]-(avg[j]**2))
    #avg[j]=0
    #std[j]=1
print(avg)
print(std)
#print(table[9])
sig=0
div=0
magic=Regression()
x=[]
y=[]
z=[]
ve = list(csv.reader(codecs.open(sys.argv[2], 'r', 'Big5')))
#for g in range(10):
"""
for k in range(0,len(table[9])-id-1,25):
    ary=[]
    for j in range(len(usage)):
        ary.extend(table[usage[j]][k+id-uslen[j]:k+id])
    for j in range(len(usage2)):
        ary.extend(numpy.power(table[usage2[j]][k+id-uslen2[j]:k+id],2))
    for j in range(len(usage3)):
        ary.extend(numpy.power(table[usage3[j]][k+id-uslen3[j]:k+id],3))
    x.append(ary)
    y.append(table[9][k+id])
    z.append(1)
for z in range(1,1000):
    magic.fit(numpy.array(x),y,z)
magic.rebuild()
"""
x=[]
y=[]
z=[]
cnt=0
for k in range(0,len(table[9])-id-1):
    if cnt+id>=(20*24):
        cnt=(cnt+1)%(20*24)
        continue
    cnt=(cnt+1)%(20*24)
    ary=[]
    """
    for j in range(len(usage)):
        ary.extend(table[usage[j]][k+id-uslen[j]:k+id])
    for j in range(len(usage2)):
        ary.extend(numpy.power(table[usage2[j]][k+id-uslen2[j]:k+id],2))
    for j in range(len(usage3)):
        ary.extend(numpy.power(table[usage3[j]][k+id-uslen3[j]:k+id],3))
    """
    for j in range(len(vlen)):
        for jj in range(len(vlen[j])):
            if vlen[j][jj]==1:
                ary.append( ((table[j][k+id-jj-1]-avg[j])/std[j]) )
                #ary.append( table[j][k+id-jj-1])
    for j in range(len(vlen2)):
        for jj in range(len(vlen2[j])):
            if vlen2[j][jj]==1:
                ary.append(((table[j][k+id-jj-1]-avg[j])/std[j])**2)
                #ary.append(table[j][k+id-jj-1]**2)
    for j in range(len(vlen3)):
        for jj in range(len(vlen3[j])):
            if vlen3[j][jj]==1:
                ary.append(((table[j][k+id-jj-1]-avg[j])/std[j])**3)
                #ary.append(table[j][k+id-jj-1]**3)
    x.append(ary)
    y.append(table[9][k+id])
    z.append(1)
"""
for z in range(1,3000):
    magic.fit(numpy.array(x),y)
magic.rebuild()
x=[]
y=[]
z=[]
"""
"""
for i in range(0, len(ve), 18):
    tmp = [[], [], [], [], [], [], [], [],
        [], [], [], [], [], [], [], [], [], []]
    for j in range(18):
        for k in range(2,len(ve[i + j])):
            try:
                float(ve[i + j][k])
                tmp[j].append(float(ve[i + j][k]))
            except ValueError:
                tmp[j].append(0.)
    for k in range(0,len(tmp[9])-id-tst):
        ary=[]
        for j in range(len(usage)):
            ary.extend(tmp[usage[j]][k+id-uslen[j]:id+k])
        for j in range(len(usage2)):
            ary.extend(numpy.power(tmp[usage2[j]][k+id-uslen2[j]:id+k],2))
        for j in range(len(usage3)):
            ary.extend(numpy.power(tmp[usage3[j]][k+id-uslen3[j]:id+k],3))
        x.append(ary)
        y.append(tmp[9][k+id])
        z.append(3)
"""
#print(len(x))
#34.0602011165 34.0602011165 0.05 n
#34.0617780172 34.0617780172 0.05 10n
#33.7869658898 33.7869658898 0.05 1
if magic.w_ is None:
    bestw=None
else:
    bestw=numpy.copy(magic.w_)
best_cost=float('Inf')

xx=[]
yy=[]
zz=[]
for rng2 in range(1,200):
    rnd=random.randint(0,len(x)-1)
    xx.append(x[rnd])
    yy.append(y[rnd])
    zz.append(z[rnd])
for h in range(1,1000):
    #print(rng)
    magic.fit(numpy.array(xx),yy,zz)
print(magic.old_cost)
magic.rebuild()

for rng in range(3000):
    xx=[]
    yy=[]
    zz=[]
    for rng2 in range(1,30):
        rnd=random.randint(0,len(x)-1)
        xx.append(x[rnd])
        yy.append(y[rnd])
        zz.append(z[rnd])
    for h in range(1,50):
        #print(rng)
        magic.fit(numpy.array(xx),yy,zz)
    print("~~~~~~~~~~~~~~~~~",magic.old_cost)
    """
    for rng2 in range(1,50):
        rnd=random.randint(0,len(x)-1)
        xx.append(x[rnd])
        yy.append(y[rnd])
        zz.append(z[rnd])
    for h in range(1,200):
        #print(rng)
        magic.fit(numpy.array(xx),yy,zz)
    #magic.rebuild()
    for h in range(1,20):
        magic.fit(numpy.array(x),y,z)
        if magic.old_cost<42:
            break
    print("~~~~~~~~~~~~~~~~~",magic.old_cost)
    for h in range(1,10):
        magic.fit(numpy.array(x),y,z)
        random.randrange
    print("~~~~~~~~~~~~~~~~~",magic.old_cost,best_cost,math.exp((best_cost-magic.old_cost)*(rng+1)/10))
    if random.random()<math.exp((best_cost-magic.old_cost)*(rng+1)):
        #print("XD")
    #if magic.old_cost <= best_cost:
        best_cost=magic.old_cost
        bestw=numpy.copy(magic.w_)
    else:
        #print("GG")
        if bestw is None:
            magic.w_=None
        else:
            magic.w_=numpy.copy(bestw)
    magic.rebuild()
    """
magic.rebuild()
if len(x)!=0:
    for h in range(1,10000):
        print(h)
        magic.fit(numpy.array(x),y,z)
        print("",magic.old_cost)
print(magic.w_)
magic.rebuild()
x=[]
y=[]
z=[]
f=codecs.open(sys.argv[3], 'w', 'Big5')
f.write("id,value\n")
sig=0
sig2=0
div=0

for i in range(0, len(ve), 18):
    qry = []
    
    for j in range(len(vlen)):
        for jj in range(len(vlen[j])):
            if vlen[j][jj]==1:
                if j == 14:
                    val=math.sin(float(ve[i+14][2+id-jj-1]))*float(ve[i+17][2+id-jj-1])
                elif j==17:
                    val=math.cos(float(ve[i+14][2+id-jj-1]))*float(ve[i+17][2+id-jj-1])
                elif j==15:
                    val=math.sin(float(ve[i+15][2+id-jj-1]))*float(ve[i+16][2+id-jj-1])
                elif j==16:
                    val=math.cos(float(ve[i+15][2+id-jj-1]))*float(ve[i+16][2+id-jj-1])
                else:
                    try:
                        val=float(ve[i+j][2+id-jj-1])
                        #qry.append(float(ve[i+j][2+id-jj-1]))
                    except ValueError:
                        val=0
                qry.append( (((val)-avg[j])/std[j]) )
    for j in range(len(vlen2)):
        for jj in range(len(vlen2[j])):
            if vlen2[j][jj]==1:
                if j == 14:
                    val=math.sin(float(ve[i+14][2+id-jj-1]))*float(ve[i+17][2+id-jj-1])
                elif j==17:
                    val=math.cos(float(ve[i+14][2+id-jj-1]))*float(ve[i+17][2+id-jj-1])
                elif j==15:
                    val=math.sin(float(ve[i+15][2+id-jj-1]))*float(ve[i+16][2+id-jj-1])
                elif j==16:
                    val=math.cos(float(ve[i+15][2+id-jj-1]))*float(ve[i+16][2+id-jj-1])
                else:
                    try:
                        val=float(ve[i+j][2+id-jj-1])
                        #qry.append(float(ve[i+j][2+id-jj-1]))
                    except ValueError:
                        val=0
                qry.append( (((val)-avg[j])/std[j]) **2)
    for j in range(len(vlen3)):
        for jj in range(len(vlen3[j])):
            if vlen3[j][jj]==1:
                if j == 14:
                    val=math.sin(float(ve[i+14][2+id-jj-1]))*float(ve[i+17][2+id-jj-1])
                elif j==17:
                    val=math.cos(float(ve[i+14][2+id-jj-1]))*float(ve[i+17][2+id-jj-1])
                elif j==15:
                    val=math.sin(float(ve[i+15][2+id-jj-1]))*float(ve[i+16][2+id-jj-1])
                elif j==16:
                    val=math.cos(float(ve[i+15][2+id-jj-1]))*float(ve[i+16][2+id-jj-1])
                else:
                    try:
                        val=float(ve[i+j][2+id-jj-1])
                        #qry.append(float(ve[i+j][2+id-jj-1]))
                    except ValueError:
                        val=0
                qry.append( (((val)-avg[j])/std[j]) **3)
    """
    for j in range(len(usage)):
        for k in range(2+id-uslen[j]+shift,2+id+shift):
            try:
                float(ve[i + usage[j]][k])
                qry.append(float(ve[i + usage[j]][k]))
            except ValueError:
                qry.append(0.)
    for j in range(len(usage2)):
        for k in range(2+id-uslen2[j]+shift,2+id+shift):
            try:
                float(ve[i + usage2[j]][k])
                qry.append(float(ve[i + usage2[j]][k])**2)
            except ValueError:
                qry.append(0.)
    for j in range(len(usage3)):
        for k in range(2+id-uslen3[j]+shift,2+id+shift):
            try:
                float(ve[i + usage3[j]][k])
                qry.append(float(ve[i + usage3[j]][k])**3)
            except ValueError:
                qry.append(0.)
    """
    ans=magic.activation(qry)
    ans=round(ans)
    f.write("id_"+str(int(i/18))+","+str(ans)+"\n")
    #sig+=abs(ans-float(ve[i + 9][id+2+shift]))**2
    #sig2+=abs(ans-float(ve[i + 9][id+2+shift]))
    #div+=1
    #print(abs(ans-float(ve[i + 9][id+2+shift])),ans,float(ve[i + 9][id+2+shift]))
#if sig/div < best:
#    best=sig/div
#    bestu=uslen[:]
#print(uslen)
#print(bestu)
#print(p,q,sig/div,sig2/div)
#print(uslen,file=result)
#print(bestu,file=result)
#print(p,q,sig/div,sig2/div,file=result)
#print(sig/div,sig2/div)
f.close()
#id=2 5.96582224071
#id=4 5.67236685864
#id=8 6.25011639749
#id=5 uasbe=all 5.20393198108
#id=5 usage=[9] 4.97530722598
#id=5 usage=[8,9] 4.75835420576
#id=5 usage=[4,5,6,8,9] 4.59735452859
#id=6 usage=[4,5,6,8,9] 4.69357493893
#id=3 usage=[4,5,6,8,9] 4.93763127199
#id=5 usage=[4,5,6,8,9,12] 4.61564952897
#id=3 usage=[4,5,6,8,9,12] 4.93860352474
#id=5 usage=[4,5,6,8,9] shift=2 4.90460092424
#id=5 usage=[0,4,5,6,8,9] shift=2 4.8557880496
#id=5 usage=[0,4,5,6,8,9,11] shift=2 4.87503022624
#id=5 usage=[0,6,8,9,16] shift=2 4.82140246372
#id=4 usage=[0,6,8,9,16] shift=2 4.73201165142
#id=3 usage=[0,6,8,9,16] shift=2 4.86558415575
#id=5 usage=[0,6,8,9,12,16] shift=3 loop=10000 4.73008048051

#0 16.6289783959
#1 15.3631528834
#2 16.4880983841
#3 19.4784197265
#4 17.134598872
#5 12.6097785164
#6 13.0510128224
#7 14.110205621
#8 9.01713468749
#9 5.16081982185
#10 20.255755226
#11 16.8842629525
#12 14.0492186206
#13 15.0919325761
#14 14.9979138787
#15 14.874133158
#16 17.6912105319
#17 18.5766859015

#9 5.16081982185
#8 9.01713468749
#5 12.6097785164
#6 13.0510128224
#7 14.1102056211
#14 14.9979138787
#15 14.874133158
#12 14.0492186206

#0 16.6289783959
#1 15.3631528834
#2 16.4880983841
#3 19.4784197265
#4 17.134598872
#10 20.255755226
#11 16.8842629525
#13 15.091932576
#16 17.6912105319
#17 18.5766859015
'''
0 1 15.638866571
0 2 15.9313932453
0 3 16.2223213035
0 4 15.3461277374
0 5 12.5389906585
0 6 12.9949957672
0 7 13.7922956514
0 8 8.62732803752
0 9 5.05948376721
0 10 16.087561792
0 11 16.1683865141
0 12 14.296598193
0 13 15.3831639144
0 14 14.5593745641
0 15 14.6857776217
0 16 16.2578820163
0 17 16.2227146148
1 2 14.8152696941
1 3 15.0916955389
1 4 14.8499733835
1 5 12.7645948336
1 6 13.048037104
1 7 13.861236373
1 8 8.65734478144
1 9 5.09003368596
1 10 14.9958631028
1 11 16.8231616032
1 12 13.9621550723
1 13 14.898332495
1 14 14.496123082
1 15 14.4948532769
1 16 15.8868539323
1 17 15.9224567906
2 3 19.7362968369
2 4 16.679201402
2 5 12.8687383789
2 6 13.1559290878
2 7 13.910735698
2 8 8.64973086361
2 9 5.11712314251
2 10 20.1407278788
2 11 16.8342843842
2 12 14.0416184176
2 13 14.5637372186
2 14 14.490596314
2 15 14.6224748696
2 16 17.0747429681
2 17 17.9675682816
3 4 17.2036364472
3 5 12.8903663175
3 6 13.1860484994
3 7 13.9407749892
3 8 8.64850307534
3 9 5.12101037446
3 10 22.8599116467
3 11 16.8506661486
3 12 14.1793450732
3 13 14.7634468049
3 14 14.5002732227
3 15 14.6101294925
3 16 17.5710298611
3 17 18.6680862131
4 5 12.8559587257
4 6 12.9568375797
4 7 13.2714404173
4 8 8.63777730657
4 9 5.02494017991
4 10 17.1449195332
4 11 16.5364506439
4 12 14.0322906533
4 13 14.6377668797
4 14 14.4822343864
4 15 14.4628363253
4 16 16.2331204787
4 17 16.7467364939
5 6 12.922423742
5 7 11.0012823666
5 8 8.59907013513
5 9 5.06910320738
5 10 12.7733783096
5 11 12.8791804447
5 12 12.3430275714
5 13 12.7648471959
5 14 12.1098400455
5 15 12.1378167521
5 16 12.674092073
5 17 12.8130864307
6 7 11.274613655
6 8 8.58032677463
6 9 5.03009582888
6 10 13.0545066309
6 11 12.9040202098
6 12 12.6832525763
6 13 13.0528573381
6 14 12.7173250683
6 15 12.7041147401
6 16 13.0582658634
6 17 13.1609438901
7 8 8.36005533006
7 9 5.04871970333
7 10 13.8887444081
7 11 13.87897361
7 12 13.0462303745
7 13 13.7844547526
7 14 13.4765704129
7 15 13.7278675255
7 16 13.9584687215
7 17 13.9837211991
8 9 5.22299397984
8 10 8.6208530877
8 11 8.51527912053
8 12 8.71329426841
8 13 8.66165118556
8 14 8.79790949683
8 15 8.60100338048
8 16 8.616898218
8 17 8.61912696309
9 10 5.11464346434
9 11 5.93626118103
9 12 5.0319214647
9 13 5.03037259719
9 14 6.55217846622
9 15 6.37554456151
9 16 5.06870464472
9 17 5.12902369189
10 11 16.8198240377
10 12 14.0380781332
10 13 14.6435597392
10 14 14.4720852118
10 15 14.6060465017
10 16 17.497367765
10 17 18.6268149646
11 12 16.0515302216
11 13 16.7963518828
11 14 14.3365338967
11 15 14.6072324178
11 16 16.8506917247
11 17 16.8750975133
12 13 13.8598915458
12 14 14.2550104706
12 15 14.5670673312
12 16 14.3677228865
12 17 14.2712124351
13 14 14.515657197
13 15 14.6310936593
13 16 15.4590414203
13 17 15.4337542196
14 15 14.4096089101
14 16 14.5187410687
14 17 14.3796348581
15 16 14.6347072047
15 17 14.6200700179
16 17 17.9241432316
'''