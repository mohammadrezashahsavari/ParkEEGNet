import tensorflow as tf
from tensorflow import Tensor
from tensorflow import transpose
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPool1D, MaxPool2D, Dense, LSTM, BatchNormalization, AveragePooling1D, ReLU, LeakyReLU, Add, Bidirectional, Concatenate, Dropout, Flatten
from tensorflow.keras.regularizers import l2
import numpy as np



def VGG_BiLSTM_Attn_Model(input_shape, n_output_nodes=1):

    inputs = Input(shape=input_shape)
    t = transpose(inputs, perm=(0, 2, 1))
    t = Conv1D(64, 3, padding='same', activation='relu')(t)
    t = Conv1D(64, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = Conv1D(128, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = Conv1D(256, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = MaxPool1D(pool_size=2, strides=2)(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    t = Conv1D(512, 3, padding='same', activation='relu')(t)
    #t = MaxPool1D(pool_size=2, strides=2)(t)

    t = Bidirectional(LSTM(64, return_sequences=True))(t)
    features, forward_state_h, forward_state_c, backward_state_h, backward_state_c = Bidirectional(LSTM(64, return_sequences=True, return_state=True))(t)
    state_h = Concatenate()([forward_state_h, backward_state_h])
    state_c = Concatenate()([forward_state_c, backward_state_c])

    context_vector, attention_weights = Attention(64)(features, state_h)

    t = Dense(256)(context_vector)
    t = LeakyReLU(alpha=0.01)(t)
    t = Dropout(0.2)(t)
    t = Dense(256)(t)
    t = LeakyReLU(alpha=0.01)(t)
    t = Dropout(0.2)(t)

    t = Dense(n_output_nodes, activation='sigmoid' if n_output_nodes == 1 else 'softmax')(t)

    return Model(inputs, t)



class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
          
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
          
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


