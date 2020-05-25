import math

import tensorflow as tf


def create(embedding_dim=50,
           n_chunks=10,
           hnet_hidden_dims=(50,),
           inner_net_dims=(784, 300, 10),
           l2reg=0):
    """
    Create `tf.keras.Model`, which takes two inputs `token` and `input_data`
    and returns outputs the predictions of labels for `input_data` as normal
    network would. Outputed model can be taught with `.fit` method, but
    learning tokens (which is needed in continual learning scenario) requires
    using custom training loop.
    
    For convinience, HyperNetwork's call method returns not only the logits,
    but also generated weights for the inner network as the second value.
    
    :param embedding_dim: dimensionality of tokens for `task` and `chunk` tokens
    :param n_chunks: HyperNetwork doesn't need to produce all the weights at
        once. To produce weights in batches, use this parameter. This is the
        number of batches per one full set of inner network's weights.
    :param hnet_hidden_dims: dimensionality of only hidden layers in
        HyperNetwork. Can be empty, e.g. `(,)`.
    :param inner_net_dims: dimensions of the inner network, including
        input_shape and output_shape, cannot be empty!
    :param l2reg: L2 regularization of the last layer in HyperNetwork
    :return: `tf.keras.Model` instance of HyperNetwork producing dense NN.
    """
    kernel_shapes = [[x, y] for x, y in zip(inner_net_dims, inner_net_dims[1:])]
    bias_shapes = [[y, ] for x, y in zip(inner_net_dims, inner_net_dims[1:])]
    weight_shapes = [[k, b] for k, b in zip(kernel_shapes, bias_shapes)]
    weight_shapes = [x for w in weight_shapes for x in w]
    # finally, weight_shapes is matrix with shapes [kernel1_shape, bias1_shape,
    #                                               kernel2_shape, bias2_shape,
    #                                               ...          , biasN_shape]
    
    weight_sizes = [tf.reduce_prod(w) for w in weight_shapes]
    weight_num = sum(weight_sizes)
    print(f"INNER NET SIZE: {weight_num}")
    
    def dense_layer(x, w, b, activation_func):
        return activation_func(tf.matmul(x, w) + b)
    
    def inner_net(weights, inputs):
        """
        Performs operations of simple feedforward network
        given all of its weights and inputs. Hardcoded relu activation!
        :param weights: `list` of weights for
                         the model in pattern
                         [weight1, bias1, weight2, bias2, ...]
        :param inputs: tensor with data [batch_size, data_dim]
        :return: network's outputs
        """
        flow = inputs
        for w, b in zip(weights[::2], weights[1::2]):
            flow = dense_layer(flow, w, b, activation_func=tf.nn.relu)
        return flow
    
    layers = [tf.keras.layers.InputLayer(input_shape=embedding_dim * 2)]
    layers += [tf.keras.layers.Dense(neurons, activation='relu')
               for neurons in hnet_hidden_dims]
    
    chunk_size = math.ceil(weight_num / n_chunks)
    layers += [tf.keras.layers.Dense(chunk_size, activation='tanh',
               kernel_regularizer=tf.keras.regularizers.l2(l2reg))]
    
    hnet = tf.keras.Sequential(layers)
    hnet.build([1, embedding_dim * 2])
    hnet.call(tf.random.uniform([1, embedding_dim * 2]))
    hnet.summary()
    
    # One network - hnet which takes tokens and outputs weights is ready here
    print(f'chunk size is {chunk_size}, there are {weight_num} params, number '
          f'of chunks is {n_chunks}')
    
    class HNet(tf.keras.Model):
        
        def __init__(self, hnet, **kwargs):
            super(HNet, self).__init__(**kwargs)
            self.hnet = hnet
            token = tf.random.normal([n_chunks, embedding_dim])
            self.chunk_tokens = self.add_weight(shape=[n_chunks, embedding_dim],
                                                trainable=True,
                                                name='chunk_embeddings',
                                                initializer='zeros')
        
        def call(self, inputs, **kwargs):
            task_token, input_data = inputs
            task_token = tf.reshape(task_token, [1, embedding_dim])
            task_token = tf.repeat(task_token, n_chunks, axis=0)
            full_token = tf.concat([self.chunk_tokens, task_token], axis=1)
            
            net_weights = self.hnet(full_token)
            net_weights_flat = tf.reshape(net_weights, (-1,))[:weight_num]
            net_weights = tf.split(net_weights_flat, weight_sizes)
            net_weights = [tf.reshape(w, shape) for w, shape in
                           zip(net_weights, weight_shapes)]
            
            output = inner_net(net_weights, input_data)
            return output, net_weights_flat
    
    full_model = HNet(hnet)
    return full_model


if __name__ == "__main__":
    model = create()
    
    import numpy as np
    
    with tf.GradientTape() as tape:
        outs, _ = model([np.random.rand(50), np.random.rand(10, 784)])
        loss = tf.reduce_sum(outs)
    grads = tape.gradient(loss, model.trainable_weights)
    print(f'gradients have been calucated without error!')
