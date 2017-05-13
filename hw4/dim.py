import numpy as np
import codecs
import math
#print(math.log(10))
np.set_printoptions(threshold=np.nan)

def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)


def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)


def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)

    return out


def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    #print(len(layers[0]),len(layers[1]))
    #print(np.shape(layers[0][0]),np.shape(layers[0][1]),np.shape(layers[1][0]),np.shape(layers[1][1]))
    gen_data = forward(data, layers)
    #print(gen_data.shape,gen_data[0].shape,gen_data[0][0])
    gen_data = gen_data @ w + b
    return gen_data

from numpy import mean,cov,double,cumsum,dot,linalg,array

def princomp(A):
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
    [latent,coeff] = linalg.eig(cov(M)) # attention:not always sorted
    score = dot(coeff.T,M) # projection of the data in the new space
    return coeff,score,latent

pre=[]
if __name__ == '__main__':
    # if we want to generate data with intrinsic dimension of 10
    mi_dim=1
    mx_dim=61
    cal_tim=30
    cal_mi=6
    cal_ma=24
    mi_hdim=60
    ma_hdim=81
    hdim_stp=5
    #N=50000
    """
    for i in range(mi_dim,mx_dim):
        pre.append([])
        dim = i
        #N=np.random.randint(10000, 100000)
        for j in range(mi_hdim,ma_hdim,hdim_stp):
            #pre[-1].append()
            lat=np.zeros(100)
            la=[[] for k in range(100)]
            for k in range(cal_tim):
                #layer_dims = [np.random.randint(60, 80), 100]
                layer_dims = [j, 100]
                A = gen_data(dim, layer_dims, np.random.randint(10000, 100000))
                #A = gen_data(dim, layer_dims, np.random.randint(100, 1000))
                coeff, score, latent = princomp(A)
                #lat=lat+latent
                for l in range(100):
                    la[l].append(latent[l])
            for l in range(100):
                la[l].sort()
                lat[l]=np.sum(la[l][cal_mi:cal_ma])
            #lat.append(latent)
            lat/=cal_ma-cal_mi
            if i!=mi_dim:
                print(lat[:3]-pre[-2][(j-mi_hdim)//hdim_stp][:3])
            pre[-1].append(lat)
            print(i,j,lat[:3])
            """
    """
            latt=lat
            lat=np.zeros(100)
            for k in range(cal_tim//30):
                #layer_dims = [np.random.randint(60, 80), 100]
                layer_dims = [j, 100]
                A = gen_data(dim, layer_dims, np.random.randint(10000, 100000))
                #A = gen_data(dim, layer_dims, np.random.randint(10, 100))
                coeff, score, latent = princomp(A)
                lat=lat+latent
            lat/=cal_tim//30
            print(i,j,lat[:3])
            print(lat[:3]-latt[:3])
            """
    """
            #print(latent)
            #print(i,j)

    np.save("mod",np.array(pre))
    """
    pre=np.load(sys.argv[1])
    data = np.load('data.npz')
    #for i in range(200):

    f=codecs.open(sys.argv[2], 'w', 'Big5')
    f.write("SetId,LogDim\n")
    an=[0 for i in range(mx_dim)]
    for i in range(200):
        A = data[str(i)]
        coeff, score, latent = princomp(A)
        print(latent)
        #print(A.shape)
        #print(coeff.shape)
        #print(score.shape)
        mi=float('Inf')
        ans=0
        for k in range(mi_dim,mx_dim):
            for j in range(mi_hdim,ma_hdim):
                val=np.sum((latent-pre[k-mi_dim][(j-mi_hdim)//hdim_stp])**2)
                #print(val)
                if(val<mi):
                    ans=k
                    mi=val
        print(ans,math.log(ans))
        f.write(str(i)+","+str(math.log(ans))+"\n")
        an[ans]+=1
    print(an)
    f.close()

            