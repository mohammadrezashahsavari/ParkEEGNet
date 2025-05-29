import tensorflow as tf
import numpy as np
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer
from keras.backend import softmax
from tensorflow.keras.layers import Input, LayerNormalization, Layer, Dense, ReLU, Dropout, Flatten, Bidirectional, LSTM, Concatenate, MaxPool1D
from tensorflow.keras.models import Model


def transfomerBasedModel(input_shape):
    
    inputs = Input(shape=input_shape)
                        # (32, 512)

    '''Aggregating information across channels using channel-wise self-attention and transformer'''
    transformer_encoder = Encoder(vocab_size=input_shape[0], sequence_length=0, h=2, d_k=64, d_v=64, d_model=512, d_ff=512, n=2, rate=0.1)
    t, self_attention_map = transformer_encoder(inputs, None, True)

    t = t + inputs

    '''Transposing for aggregating information along the time axis'''
    t = transpose(t, perm=(0, 2, 1))

    t = MaxPool1D(pool_size=2, strides=2)(t)

    '''Aggregating information along the time axis using bidirectional GRU layers and additive attention mechanism'''
    t = Bidirectional(LSTM(64, return_sequences=True))(t)
    features, forward_state_h, forward_state_c, backward_state_h, backward_state_c = Bidirectional(LSTM(64, return_sequences=True, return_state=True))(t)
    state_h = Concatenate()([forward_state_h, backward_state_h])
    state_c = Concatenate()([forward_state_c, backward_state_c])

    context_vector, attention_weights = AdditiveAttention(64)(features, state_h)

    #t = Flatten()(t)
    t = Dense(256, activation='relu')(context_vector)
    t = Dense(256, activation='relu')(t)
    t = Dense(1, activation='sigmoid')(t)
    return Model(inputs, t)



class AdditiveAttention(tf.keras.Model):
    def __init__(self, units):
        super(AdditiveAttention, self).__init__()
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





'''
class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)   
        position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)                                          
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim,
            weights=[word_embedding_matrix],
            trainable=False
        )
        self.position_embedding_layer = Embedding(
            input_dim=sequence_length, output_dim=output_dim,
            weights=[position_embedding_matrix],
            trainable=False
        )
             
    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P
 
 
    def call(self, inputs):        
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices
'''


# Implementing the Scaled-Dot Product Attention
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation
        weights = softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)

# Implementing the Multi-Head Attention
class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = h  # Number of attention heads
        self.d_k = d_k  # Dimension of queries and keys
        self.d_v = d_v  # Dimension of values
        self.d_model = d_model  # Model dimension
        self.W_q = Dense(d_k)  # Linear projection for queries
        self.W_k = Dense(d_k)  # Linear projection for keys
        self.W_v = Dense(d_v)  # Linear projection for values
        self.W_o = Dense(d_model)  # Linear projection for final output

    def call(self, queries, keys, values, mask=None):
        # Project queries, keys, and values
        q_reshaped = self.W_q(queries)  # (batch_size, seq_len, d_k)
        k_reshaped = self.W_k(keys)  # (batch_size, seq_len, d_k)
        v_reshaped = self.W_v(values)  # (batch_size, seq_len, d_v)

        # Compute attention scores
        scores = matmul(q_reshaped, k_reshaped, transpose_b=True) / tf.math.sqrt(float(self.d_k))

        # Apply mask if present
        if mask is not None:
            scores += -1e9 * mask

        # Compute attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Compute weighted values
        output = matmul(attention_weights, v_reshaped)

        # Final projection
        output = self.W_o(output)

        return output, attention_weights  # Return both output and attention weights

    

# Implementing the Add & Norm Layer
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer

    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + sublayer_x

        # Apply layer normalization to the sum
        return self.layer_norm(add)

# Implementing the Feed-Forward Layer
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer

    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        x_fc1 = self.fully_connected1(x)

        return self.fully_connected2(self.activation(x_fc1))

# Implementing the Encoder Layer
class EncoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def call(self, x, padding_mask, training):
        # Multi-head attention layer (returns output + attention weights)
        multihead_output, attention_weights = self.multihead_attention(x, x, x, padding_mask)

        # Apply dropout
        multihead_output = self.dropout1(multihead_output, training=training)

        # Add & Norm
        addnorm_output = self.add_norm1(x, multihead_output)

        # Feed-forward
        feedforward_output = self.feed_forward(addnorm_output)

        # Dropout + Add & Norm
        feedforward_output = self.dropout2(feedforward_output, training=training)
        final_output = self.add_norm2(addnorm_output, feedforward_output)

        return final_output, attention_weights  # Return both final output and attention weights


# Implementing the Encoder
class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dropout = Dropout(rate)
        self.encoder_layers = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

    def call(self, input_sentence, padding_mask, training):
        x = input_sentence
        attention_maps = []  # Store attention maps for all layers

        for layer in self.encoder_layers:
            x, attn = layer(x, padding_mask, training)
            attention_maps.append(attn)

        return x, attention_maps  # Return final output and attention maps
