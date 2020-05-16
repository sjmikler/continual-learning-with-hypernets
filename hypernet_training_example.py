# %%

import tensorflow_datasets as tfds
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

model.build([1, 784])
model.compile(optimizer, loss_fn)
model.summary()

# such model has 79,510 parameters, let's try and generate them
# we won't be albe to generate its weights while keeping the whole
# differentiable, but we will create identical model in different way

# To get identical model later, I keep all shapes from this model

weights_shapes = [x.shape for x in model.weights]
weights_sizes = [tf.reduce_prod(x.shape) for x in model.weights]

# %%

weight_generator = tf.keras.Sequential([
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(79510)
])

# we need to choose size of task encoding, e.g. 50
weight_generator.build([1, 50])
weight_generator.summary()

# model generating this small network is very large, it has 23,947,810 params

# %%

def dense(x, w, b, activation_func):
    return activation_func(tf.matmul(x, w) + b)


def inner_model(weights, inputs):
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
        flow = dense(flow, w, b, activation_func=tf.nn.relu)
    return flow


# %%

token = tf.keras.layers.Input(shape=[50], batch_size=1)
data = tf.keras.layers.Input(shape=[784])

weights = weight_generator(token)
weights = tf.squeeze(weights)
split_weights = tf.split(weights, weights_sizes, axis=0)
split_weights = [tf.reshape(weight, shape) for weight, shape in
                 zip(split_weights, weights_shapes)]

outputs = inner_model(split_weights, data)
joint_model = tf.keras.Model(inputs=[token, data], outputs=outputs)

# %%

rnd_token = tf.random.uniform([1, 50])
rnd_data = tf.random.uniform([10, 784])
test_output = joint_model([rnd_token, rnd_data])

# %%

ds = tfds.load(name='mnist', as_supervised=True)

def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, shape=[-1])
    x /= 255
    return x, y


def attach_token(x, y):
    global rnd_token
    return (rnd_token, x), y


ds['train'] = ds['train'].repeat().shuffle(20000).map(preprocess).batch(128)
ds['test'] = ds['test'].map(preprocess).batch(128)

full_train_ds = ds['train'].map(attach_token)
full_test_ds = ds['test'].map(attach_token)

# %%

loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
joint_model.compile(optimizer, loss_fn, metrics=['accuracy'])

joint_model.fit(x=full_train_ds, validation_data=full_test_ds,
                steps_per_epoch=2000, epochs=20)

# It works! But it is weak, as it achieves ~70% accuracy.

# %%
