from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM, Flatten, Conv1D, LocallyConnected1D, CuDNNLSTM, CuDNNGRU, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from math import sqrt
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
# from matplotlib import pyplot
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import ELU
import tensorflow as tf
import numpy as np
import argparse
import os
from keras.callbacks import CSVLogger
from keras import backend as K
from keras import layers
from attention import *

class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
                super().__init__()
                self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
                self.ffn = keras.Sequential(
                    [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
                )
                self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = layers.Dropout(rate)
                self.dropout2 = layers.Dropout(rate)

        def call(self, inputs, training):
                attn_output = self.att(inputs, inputs)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
        def __init__(self, maxlen, vocab_size, embed_dim):
                super().__init__()
                self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
                self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

        def call(self, x):
                maxlen = tf.shape(x)[-1]
                positions = tf.range(start=0, limit=maxlen, delta=1)
                positions = self.pos_emb(positions)
                x = self.token_emb(x)
                return x + positions

def Trsf(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(TokenAndPositionEmbedding(alphabet_size, alphabet_size, embed_dim=32))
        model.add(TransformerBlock(embed_dim=32, num_heads=2, ff_dim=32))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.1))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model


def biGRU(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Bidirectional(CuDNNGRU(32, stateful=False, return_sequences=True)))
        model.add(Bidirectional(CuDNNGRU(32, stateful=False, return_sequences=False)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def biGRU_big(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Bidirectional(CuDNNGRU(128, stateful=False, return_sequences=True)))
        model.add(Bidirectional(CuDNNGRU(128, stateful=False, return_sequences=False)))
#        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def biGRU_16bit(bs,time_steps,alphabet_size):
        K.set_floatx('float16')
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Bidirectional(CuDNNGRU(32, stateful=False, return_sequences=True)))
        model.add(Bidirectional(CuDNNGRU(32, stateful=False, return_sequences=False)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def biLSTM(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=True)))
        model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=False)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model


def biLSTM_16bit(bs,time_steps,alphabet_size):
        K.set_floatx('float16')
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=True)))
        model.add(Bidirectional(CuDNNLSTM(32, stateful=False, return_sequences=False)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def LSTM_multi(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
        model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def LSTM_multi_big(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 64, batch_input_shape=(bs, time_steps)))
        model.add(CuDNNLSTM(64, stateful=False, return_sequences=True))
        model.add(CuDNNLSTM(64, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def LSTM_multi_bn(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
        model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def LSTM_multi_16bit(bs,time_steps,alphabet_size):
        K.set_floatx('float16')
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
        model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def LSTM_multi_selu(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
        model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(64, activation=keras.activations.selu, kernel_initializer=init))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def LSTM_multi_selu_16bit(bs,time_steps,alphabet_size):
        K.set_floatx('float16')
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
        model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        init = keras.initializers.lecun_uniform(seed=0)
        model.add(Dense(64, activation=keras.activations.selu, kernel_initializer=init))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def GRU_multi(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(CuDNNGRU(32, stateful=False, return_sequences=True))
        model.add(CuDNNGRU(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def GRU_multi_big(bs,time_steps,alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(CuDNNGRU(128, stateful=False, return_sequences=True))
        model.add(CuDNNGRU(128, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def GRU_multi_16bit(bs,time_steps,alphabet_size):
        K.set_floatx('float16')
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(CuDNNGRU(32, stateful=False, return_sequences=True))
        model.add(CuDNNGRU(32, stateful=False, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model




def FC_4layer_16bit(bs,time_steps, alphabet_size):
        K.set_floatx('float16')
        model = Sequential()
        model.add(Embedding(alphabet_size, 5, batch_input_shape=(bs, time_steps)))
        model.add(Flatten())
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def FC_4layer(bs,time_steps, alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 5, batch_input_shape=(bs, time_steps)))
        model.add(Flatten())
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def FC_4layer_big(bs,time_steps, alphabet_size):
        model = Sequential()
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Flatten())
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(128, activation=ELU(1.0)))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

def FC_16bit(bs,time_steps,alphabet_size):
        k.set_floatx('float16')
        model = Sequential()
        init = keras.initializers.lecun_uniform(seed=0)
        model.add(embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(flatten())
        model.add(dense(1024, activation='relu', kernel_initializer=init))
        model.add(dense(64, activation='relu', kernel_initializer=init))
        model.add(dense(alphabet_size, activation='softmax'))
        return model


def FC(bs,time_steps,alphabet_size):
        model = Sequential()
        init = keras.initializers.lecun_uniform(seed=0)
        model.add(Embedding(alphabet_size, 32, batch_input_shape=(bs, time_steps)))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu', kernel_initializer=init))
        model.add(Dense(64, activation='relu', kernel_initializer=init))
        model.add(Dense(alphabet_size, activation='softmax'))
        return model

