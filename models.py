import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Merge, Reshape, Dropout, Convolution2D, MaxPooling2D, ZeroPadding2D, Flatten

def vis_lstm(dim1):

    embedding_model = Sequential()
    embedding_model.add(Embedding(
        dim1,
        512,


        trainable=False))



    image_model = Sequential()
    image_model.add(Dense(
        512,
        input_dim=4096,
        activation='linear'))


    main_model = Sequential()
    main_model.add(Merge(
        [image_model, embedding_model],
        mode='concat',
        concat_axis=1))


    main_model.add(LSTM(1001))
    main_model.add(Dropout(0.5))
    main_model.add(Dense(1000, activation='softmax'))
    print main_model.summary()

    return main_model
