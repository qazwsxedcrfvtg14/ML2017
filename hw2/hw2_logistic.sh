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
mitst=0
mini=0
#for tst in range(18,106):
for tst in range(1):
    #random.seed(str([7122,59491,66666,233333,9487]))
    random.seed(7122)
    best=float('Inf')
    class Regression(object):
        def __init__ (self):
            #self.w_ = numpy.zeros(1 + 18*id)
            self.w_ = None #numpy.zeros(1 + id*len(usage))
            self.g_ = None
            self.old_cost=float('Inf')
            self.eta=3
            self.cnt=0
            self.cost=float('Inf')
        def rebuild(self):
            self.g_ = None
            self.old_cost=float('Inf')
            self.cost=float('Inf')
            #self.eta=0
        def fit(self,x,y,z):
            if self.w_ is None:
                self.w_=numpy.zeros(len(x[0]))
                #self.w_=numpy.full(len(x[0]),1.0)
                #self.w_=numpy.zeros(len(x[0]))
                #print("a")
                #self.w_=numpy.random.rand(len(x[0])+1,1)*1e-3
                #print("b")
            output = numpy.dot(x,self.w_)

            output=1.0/(numpy.add(numpy.exp(-output),1))

            errors = y - output 
            self.cnt+=1
            #if self.cnt % 30 == 0:
            self.cost=numpy.sum(numpy.abs(errors))/len(x)


            self.old_cost=self.cost
            g=x.T.dot(errors)
            
            #g+=(-1e1)*self.w_/len(x)

            if self.g_ is None:
                self.g_=numpy.full(len(g),1e-30)
            self.g_+=numpy.power(g,2)
            g/=numpy.sqrt(self.g_)
            

            """
            print(g)
            print(t)
            print(self.eta*g)
            print(self.eta*t)
            """
            self.w_+=self.eta*g
            return self
        def activation(self,x):
            val=-numpy.dot(x,self.w_)
            if val>500:
                val=500
            return 1.0/(1+math.exp(val))
    #from scipy import linspace, polyval, polyfit, sqrt, stats, randn
    #from pylab import plot, title, show , legend
    x = []
    y = []
    z = []
    ve = list(csv.reader(codecs.open(sys.argv[3], 'r', 'Big5')))
    ve2 = list(csv.reader(codecs.open(sys.argv[4], 'r', 'Big5')))
    #fun=list(range(106))
    #del fun[22*4]
    #del fun[tst]
    #del fun[32]
    #"""
    #del fun[51]
    #del fun[44]
    #del fun[12]
    #del fun[8]
    #del fun[7]
    #"""
    #del fun[2*4]
    """
    sq=[0,1,3,4,5]
    sq2=[0,1,3,4,5]
    lq=[0,1,3,4,5]
    lq2=[0]
    lq3=[0,0,0,0,1,1,1,3,3,4]
    lq4=[1,3,4,5,3,4,5,4,5,5]
    lq5=[1,3,4,5,3,4,5,4,5,5]
    """

    fun=list(range(106)) 
    del fun[51] 
    del fun[44] 
    del fun[12] 
    del fun[8] 
    del fun[7] 
    sq=[0,1,3,4,5] 
    sq2=[1] 
    lq=[3,5] 
    lq2=[0] 
    lq2_rat=-1 
    lq3=[]
    lq4=[]
    lq5=[]

    std=[0 for x in range(len(fun)+len(sq)+len(sq2)+len(lq)+len(lq2)+len(lq3)+len(lq5))]
    avg=[0 for x in range(len(fun)+len(sq)+len(sq2)+len(lq)+len(lq2)+len(lq3)+len(lq5))]
    y_tot=0
    for i in range(1, len(ve)):
        try:
            val=float(ve2[i-1][0])
        except ValueError:
            val=0.
        ary=[]
        for j in range(len(fun)):
            val2=float(ve[i][fun[j]])
            std[j]+=val2**2
            avg[j]+=val2
            ary.append(val2)
        for j in range(len(sq)):
            val2=float(ve[i][sq[j]])*float(ve[i][sq[j]])
            std[j+len(fun)]+=val2**2
            avg[j+len(fun)]+=val2
            ary.append(val2)
        for j in range(len(sq2)):
            val2=float(ve[i][sq2[j]])*float(ve[i][sq2[j]])*float(ve[i][sq2[j]])
            std[j+len(fun)+len(sq)]+=val2**2
            avg[j+len(fun)+len(sq)]+=val2
            ary.append(val2)
        for j in range(len(lq)):
            val2=math.sqrt(float(ve[i][lq[j]]))
            std[j+len(fun)+len(sq)+len(sq2)]+=val2**2
            avg[j+len(fun)+len(sq)+len(sq2)]+=val2
            ary.append(val2)
        for j in range(len(lq2)):
            val2=math.exp(float(ve[i][lq2[j]]))
            std[j+len(fun)+len(sq)+len(sq2)+len(lq)]+=val2**2
            avg[j+len(fun)+len(sq)+len(sq2)+len(lq)]+=val2
            ary.append(val2)
        for j in range(len(lq3)):
            val2=float(ve[i][lq3[j]])*float(ve[i][lq4[j]])
            std[j+len(fun)+len(sq)+len(sq2)+len(lq)+len(lq2)]+=val2**2
            avg[j+len(fun)+len(sq)+len(sq2)+len(lq)+len(lq2)]+=val2
            ary.append(val2)
        for j in range(len(lq5)):
            val2=math.tanh(float(ve[i][lq5[j]]))
            std[j+len(fun)+len(sq)+len(sq2)+len(lq)+len(lq2)+len(lq3)]+=val2**2
            avg[j+len(fun)+len(sq)+len(sq2)+len(lq)+len(lq2)+len(lq3)]+=val2
            ary.append(val2)
        ary.append(1)
        y_tot+=val
        y.append(val)
        x.append(ary)
    y_avg=y_tot/(len(ve)-1)
    for j in range(len(std)):
        avg[j]=avg[j]/(len(ve)-1)
        std[j]=math.sqrt(std[j]/(len(ve)-1)-(avg[j]**2))
        if std[j]<=1:
            avg[j]=0
            std[j]=1
        #avg[j]=0
        #std[j]=1
    #print(avg,std)
    for i in range(len(x)):
        for j in range(len(std)):
            x[i][j]=((x[i][j]-avg[j])/std[j])
    #print(x)
    print("init")
    magic=Regression()
    magic.fit(numpy.array(x),y,z)
    """
    for k in range(5):
        for j in range(2):
            rng=list(range(20))
            #random.shuffle(rng)
            for i in rng:
                xx=[]
                yy=[]
                zz=[]
                for rng2 in range(1+i,len(x),20):
                    xx.append(x[rng2])
                    yy.append(y[rng2])
                for h in range(1,5):
                    print(k,i,h)
                    magic.fit(numpy.array(xx),yy,zz)
                    print(magic.old_cost)
        magic.eta*=0.7
    print(magic.old_cost)
    print(magic.w_)
    magic.rebuild()
    #magic.w_[1]=0
    """
    x2 = []
    cut=-1

    for i in range(len(x)):
        x2.append([])
        for j in range(len(fun)):
            if math.fabs(magic.w_[j])<=cut:
                continue
            x2[-1].append(x[i][j])
        for j in range(len(sq)):
            x2[-1].append(x[i][len(fun)+j])
        for j in range(len(sq2)):
            x2[-1].append(x[i][len(fun)+len(sq)+j])
        for j in range(len(lq)):
            x2[-1].append(x[i][len(fun)+len(sq)+len(sq2)+j])
        for j in range(len(lq2)):
            x2[-1].append(x[i][len(fun)+len(sq)+len(sq2)+len(lq)+j])
        for j in range(len(lq3)):
            x2[-1].append(x[i][len(fun)+len(sq)+len(sq2)+len(lq)+len(lq2)+j])
        for j in range(len(lq5)):
            x2[-1].append(x[i][len(fun)+len(sq)+len(sq2)+len(lq)+len(lq2)+len(lq3)+j])
        x2[-1].append(1)

    magic2=Regression()
    for k in range(5):
        for j in range(2):
            rng=list(range(20))
            random.shuffle(rng)
            for i in rng:
                xx=[]
                yy=[]
                zz=[]
                for rng2 in range(1+i,len(x),20):
                    xx.append(x2[rng2])
                    yy.append(y[rng2])
                for h in range(1,20):
                    print(k,i,h)
                    magic2.fit(numpy.array(xx),yy,zz)
                    print(magic2.old_cost)
        magic2.eta*=0.7122
    #tmp=magic2.eta
    #magic2.rebuild()
    #magic2.eta=1e-3
    for h in range(1,10):
        print(h)
        magic2.fit(numpy.array(x2),y,z)
        print(magic2.old_cost)

    #print(tmp)
    #magic2.rebuild()
    """
    magic2.eta=1
    for h in range(1,10):
        print(h)
        magic2.fit(numpy.array(x2),y,z)
        print(magic2.old_cost)
    magic2.rebuild()
    magic2.eta=0.01
    for k in range(10):
        for h in range(1,200):
            print(h)
            magic2.fit(numpy.array(x2),y,z)
            print(magic2.old_cost)
        tmp=magic2.eta/2
        magic2.rebuild()
        magic2.eta=tmp
    """
    """
    magic2.rebuild()
    magic2.eta=1e-6
    for h in range(1,2):
        print(h)
        magic2.fit(numpy.array(x2),y,z)
        print(magic2.old_cost)
    """
    f=codecs.open(sys.argv[6], 'w', 'Big5')
    f.write("id,label\n")
    sig=0
    sig2=0
    div=0
    print(magic2.w_)
    ve = list(csv.reader(codecs.open(sys.argv[5], 'r', 'Big5')))
    right=0
    #print(len(x2[0]))
    for i in range(len(x2)):
        ans=magic2.activation(x2[i])
        if ans>0.5:
            ans=1
        else:
            ans=0
        if ans == y[i]:
            right+=1
    print(right,len(x2),right/len(x2))
    right=0
    for i in range(1, len(ve)):
        qry = []
        for j in range(len(fun)):
            if math.fabs(magic.w_[j])<=cut:
                continue
            val=float(ve[i][fun[j]])
            val=((val-avg[j])/std[j])
            qry.append(val)
            
        for j in range(len(sq)):
            val=float(ve[i][sq[j]])*float(ve[i][sq[j]])
            val=((val-avg[j+len(fun)])/std[j+len(fun)])
            qry.append(val)
            
        for j in range(len(sq2)):
            val=float(ve[i][sq2[j]])*float(ve[i][sq2[j]])*float(ve[i][sq2[j]])
            val=((val-avg[j+len(fun)+len(sq)])/std[j+len(fun)+len(sq)])
            qry.append(val)

        for j in range(len(lq)):
            val=math.sqrt(float(ve[i][lq[j]]))
            val=((val-avg[j+len(fun)+len(sq)+len(sq2)])/std[j+len(fun)+len(sq)+len(sq2)])
            qry.append(val)

        for j in range(len(lq2)):
            val=math.exp(float(ve[i][lq2[j]]))
            val=((val-avg[j+len(fun)+len(sq)+len(sq2)+len(lq)])/std[j+len(fun)+len(sq)+len(sq2)+len(lq)])
            qry.append(val)

        for j in range(len(lq3)):
            val=float(ve[i][lq3[j]])*float(ve[i][lq4[j]])
            val=((val-avg[j+len(fun)+len(sq)+len(sq2)+len(lq)+len(lq2)])/std[j+len(fun)+len(sq)+len(sq2)+len(lq)+len(lq2)])
            qry.append(val)

        for j in range(len(lq5)):
            val=math.tanh(float(ve[i][lq5[j]]))
            val=((val-avg[j+len(fun)+len(sq)+len(sq2)+len(lq)+len(lq2)+len(lq3)])/std[j+len(fun)+len(sq)+len(sq2)+len(lq)+len(lq2)+len(lq3)])
            qry.append(val)

        qry.append(1)
        ans=magic2.activation(qry)
        #ans=round(ans)
        #if ans>(1-y_avg):
        if ans>0.5:
            f.write(str(int(i))+",1\n")
        else :
            f.write(str(int(i))+",0\n")
    print(right,(len(ve)-1),right/(len(ve)-1))
    f.close()
    if right/(len(ve)-1)>mini:
        mini=right/(len(ve)-1)
        mitst=tst
    print(tst,mitst)
quit()
while True:
    try:
        eval(input())
    except:
        pass