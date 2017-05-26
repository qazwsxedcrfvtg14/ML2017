import numpy as np
import string
import sys
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.layers import MaxPooling1D,Convolution1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
#print(lancaster_stemmer.stem('maximum'))
#quit()


from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.55
set_session(tf.Session(config=config))
np.set_printoptions(threshold=np.nan)
#train_path = sys.argv[1]
#test_path = sys.argv[2]
#output_path = sys.argv[3]

test_path = sys.argv[1]
output_path = sys.argv[2]

#####################
###   parameter   ###
#####################
split_ratio = 0
embedding_dim = 100
nb_epoch = 1000
batch_size = 128


################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r',encoding='utf-8') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

###########################
###   custom metrices   ###
###########################

def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))
    
#########################
###   Main function   ###
#########################
def main():

    ### read training and testing data
    
    (_, X_data,_) = read_data(test_path,False)
    for i in range(len(X_data)):
        #X_data[i]=(" ".join([x for x in X_data[i].split() if x not in stop]))
        X_data[i]=X_data[i].lower()
        X_data[i]="".join([x for x in X_data[i] if( ord(x) >= ord("a") and ord(x) <= ord("z") ) or ord(x) == ord(" ")])
        X_data[i]=(" ".join([x for x in X_data[i].split()]))
    
    #pickle.dump(tokenizer,open('tokenizer','wb'),True)
    tokenizer = pickle.load(open('tokenizer','rb'))
    tag_list = pickle.load(open('tag_list','rb'))
    word_index = tokenizer.word_index
    ### convert word sequences to index sequence
    print ('Convert to index sequences.')
    test_sequences = tokenizer.texts_to_sequences(X_data)
    #test_sequences = tokenizer.texts_to_sequences(X_test)
    W=30000

    test_sequences_=[]
    for i in test_sequences:
        test_sequences_.append(np.array([0 for i in range(W)]))
        for j in i:
            test_sequences_[-1][j]+=1
    test_sequences=np.array(test_sequences_)
    model=load_model("my_model",custom_objects={'f1_score': f1_score})
    #model.save("my_model")
    #model=load_model("best_word_TA.hdf5",custom_objects={'f1_score': f1_score})
    #print(Y_pred)
    thresh = 0.5
    Y_pred = model.predict(test_sequences)
    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        for index,labels in enumerate(Y_pred):
            lab=labels
            labels = [tag_list[i] for i,value in enumerate(labels) if value>thresh ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
    quit()
if __name__=='__main__':
    main()
