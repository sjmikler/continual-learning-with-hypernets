import math
import tensorflow as tf


def create(embedding_dim=50,
           n_chunks=1,
           hnet_hidden_dims=(1000, 1000),
           inner_net_dims=(784, 300, 10)):
    """
    Create `tf.keras.Model`, which takes two arguments `token` and `input_data`
    and returns outputs of the inner network.
    
    :param n_chunks:
    :param embedding_dim: embedding size of the `token`
    :param hnet_hidden_dims: hidden dimensions of hypernetwork
    :param inner_net_dims: all dimensions of the inner network, including
                           input_shape and output_shape
    :return: `tf.keras.Model` instance
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
    layers += [tf.keras.layers.Dense(chunk_size)]
    
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
            net_weights = tf.reshape(net_weights, (-1,))[:weight_num]
            net_weights = tf.split(net_weights, weight_sizes)
            net_weights = [tf.reshape(w, shape) for w, shape in
                           zip(net_weights, weight_shapes)]
            
            output = inner_net(net_weights, input_data)
            return output
    
    full_model = HNet(hnet)
    return full_model


if __name__ == "__main__":
    model = create()
    
    import numpy as np
    
    with tf.GradientTape() as tape:
        outs = model([np.random.rand(50), np.random.rand(10, 784)])
        loss = tf.reduce_sum(outs)
    grads = tape.gradient(loss, model.trainable_weights)
    print(f'gradients have been calucated without error!')
