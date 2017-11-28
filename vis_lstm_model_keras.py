import numpy as np
import embedding
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Merge, Reshape, Dropout, Convolution2D, MaxPooling2D, ZeroPadding2D, Flatten

def build_model(self,options):

    embedding_model = Sequential()
    embedding_model.add(Embedding(
        options['q_vocab_size'],
        options['embedding_size'],

        trainable=True))



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
    main_model.add(Dropout(self.options['word_emb_dropout']))
    main_model.add(Dense(1001, activation='softmax'))

    return input_tensors, loss, accuracy, predictions
