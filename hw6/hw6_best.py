#coding = utf8
# 1.41145
import pandas as pd
import numpy as np
from IPython import embed
def load_data( tpath):

    test_data = pd.read_csv(tpath)
    test_data.UserID = test_data.UserID.astype('category')
    test_data.MovieID = test_data.MovieID.astype('category')

    return test_data

import keras.models as kmodel
import keras.backend as K
import keras
from sklearn import cross_validation
import tensorflow as tf
from keras import regularizers
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
def generate_model(n_movies, n_users):
        movie_input = keras.layers.Input(shape=(1,))
        #movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 120,embeddings_regularizer=regularizers.l2(1e-2))(movie_input))
        movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 200, embeddings_initializer='random_uniform')(movie_input))
        movie_vec = keras.layers.Dropout(0.3)(movie_vec)
        user_input = keras.layers.Input(shape=(1,))
        #user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 120,embeddings_regularizer=regularizers.l2(1e-2))(user_input))
        user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 200, embeddings_initializer='random_uniform')(user_input))
        user_vec = keras.layers.Dropout(0.3)(user_vec)
        input_vecs = keras.layers.merge([movie_vec,user_vec],'dot') 
        #input_vecs = keras.layers.Dropout(0.2)(input_vecs)
        #input_vecs = keras.layers.Dense(5, activation='softmax')(input_vecs)
        model = kmodel.Model([movie_input, user_input], input_vecs)
        #model.compile(optimizer = 'adam',loss = 'categorical_crossentropy')
        model.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics=[root_mean_squared_error])
        #model.summary()
        return model
if __name__ == '__main__':
    test = load_data(sys.argv[1]+'test.csv')
    m_test = test.MovieID
    u_test = test.UserID
    model = load_model("gg.h5py",custom_objects={'root_mean_squared_error': root_mean_squared_error})

    #y_pred =np.argmax( model.predict([movieid,userid]),1)+1
    y_pred = np.clip(model.predict([m_test,u_test]),1,5)
    #y_pred = model.predict([m_test,u_test])
    #model.save('gg.h5py')
    output = pd.DataFrame({'TestDataID':np.arange(len(y_pred)+1)[1:],'Rating':y_pred.flatten()}, columns = ['TestDataID','Rating'])   
    output.to_csv(sys.argv[2],index=False)
