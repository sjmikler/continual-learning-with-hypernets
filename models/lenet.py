import tensorflow as tf


def dense(*args, **kwargs):
    return tf.keras.layers.Dense(*args,
                                 **kwargs,
                                 kernel_initializer=gl_initializer,
                                 kernel_regularizer=gl_regularizer)


def conv(*args, **kwargs):
    return tf.keras.layers.Conv2D(*args,
                                  **kwargs,
                                  kernel_initializer=gl_initializer,
                                  kernel_regularizer=gl_regularizer)


def LeNet(input_shape,
          n_classes,
          l2_reg=0,
          layer_sizes=(300, 100),
          initializer='glorot_uniform',
          **kwargs):
    global gl_regularizer, gl_initializer
    gl_regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    gl_initializer = initializer
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    flow = tf.keras.layers.Flatten()(inputs)
    
    for layer_size in layer_sizes:
        flow = dense(layer_size, activation='relu')(flow)
    
    outs = dense(n_classes, activation=None)(flow)
    model = tf.keras.Model(inputs=inputs, outputs=outs)
    return model


def LeNetConv(input_shape, n_classes, l2_reg=0, initializer='glorot_uniform',
              **kwargs):
    global gl_regularizer, gl_initializer
    gl_regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    gl_initializer = initializer
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    flow = conv(20, 5, activation='relu')(inputs)
    flow = tf.keras.layers.MaxPool2D(2)(flow)
    flow = conv(50, 5, activation='relu')(flow)
    flow = tf.keras.layers.MaxPool2D(2)(flow)
    
    flow = tf.keras.layers.Flatten()(flow)
    flow = dense(500, activation='relu')(flow)
    
    outs = dense(n_classes, activation=None)(flow)
    model = tf.keras.Model(inputs=inputs, outputs=outs)
    return model
