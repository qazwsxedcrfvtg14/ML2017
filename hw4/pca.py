from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def princomp(A):
    # computing eigenvalues and eigenvectors of covariance matrix
    #mean = A.mean(axis=0)
    M = (A).T # subtract the mean (along columns)
    [latent,coeff] = np.linalg.eigh(np.cov(M)) # attention:not always sorted
    score = np.dot(coeff.T,M) # projection of the data in the new space
    return coeff,score,latent

arr = np.array([])
for c in range(10):
    for i in range(10):
        im=Image.open("img/"+chr(c+ord('A'))+"0"+chr(i+ord('0'))+".bmp",)
        img = np.array(im).flatten().astype('float64')
        arr=np.append(arr,img)
arr=arr.reshape(100,64*64)
mean = arr.mean(axis=0)
arr-=mean

#U,s,V=np.linalg.svd(arr.T)
C,S,L=princomp(arr)
#idx=L.argsort()[::-1]
U=(C.T)[::-1]
#U+=mean

#p1-1
print(mean.shape)
fg=plt.figure()
ax=fg.add_subplot(1,1,1)
ax.imshow(mean.reshape(64,64),cmap='gray')
plt.xticks(np.array([]))
plt.yticks(np.array([]))
fg.suptitle('AVG')
fg.savefig('1.1o.png')

fg=plt.figure()
for i in range(9): 
    ax=fg.add_subplot(3,3,i+1)
    ax.imshow(U[i].reshape(64,64),cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
fg.suptitle('Eigenface')
fg.savefig('1.1.png')

#p1-2
res=[]
for i in range(100):
    #print(U[:5].shape,arr[i].T.reshape(64*64,1).shape)
    ot=np.matmul(U[:5],arr[i].T).T
    #print(ot)
    #ot=np.matmul(ot,U[:5])
    ot = U[0]*ot[0] + U[1]*ot[1] + U[2]*ot[2] + U[3]*ot[3] + U[4]*ot[4]
    #ot=np.matmul([0.2,0.2,0.2,0.2,0.2],U[:5])
    #print(ot)
    res.append(ot+mean)
    #res.append(arr[i])

fg=plt.figure()
for i in range(100):
    ax=fg.add_subplot(10,10,i+1)
    ax.imshow(arr[i].reshape(64,64),cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
fg.suptitle('source')
fg.savefig('1.2a.png')

fg=plt.figure()
for i in range(100):
    ax=fg.add_subplot(10,10,i+1)
    ax.imshow(res[i].reshape(64,64),cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
fg.suptitle('reconstruct')
fg.savefig('1.2b.png')

#p1-3
ans = 0
for k in range(1,100):
    res=[]
    for i in range(100):
        ot=np.matmul(U[:k],arr[i].T).T
        #print(ot)
        ot=np.matmul(ot,U[:k]).reshape(64*64)
        #ot = U[0]*ot[0] + U[1]*ot[1] + U[2]*ot[2] + U[3]*ot[3] + U[4]*ot[4]
        res.append(ot)
    res=np.array(res)
    rmse=np.sqrt(sum(sum((res-arr)**2))/(100*64*64))/255
    print(rmse)
    if rmse <= 0.01:
        ans = k
        break
print(ans)





