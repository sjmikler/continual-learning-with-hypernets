# %%

import tensorflow as tf
import tensorflow_datasets as tfds

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10)
])

loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

model.build([1, 784])
model.compile(optimizer, loss_fn)
model.summary()

# such model has not many parameters, let's try and generate them
# we won't be albe to generate its weights while keeping the whole
# differentiable, but we will create identical model in different way

# To get identical model later, I keep all shapes from this model

weights_shapes = [x.shape for x in model.weights]
weights_sizes = [tf.reduce_prod(x.shape) for x in model.weights]

# %%

weight_generator = tf.keras.Sequential([
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(7850)
])

# we need to choose size of task encoding, e.g. 50
weight_generator.build([1, 50])
weight_generator.summary()


# model generating this small network is very large, it has millions parameters
# we will generate all at once, in next versions we will generate chunks

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

mnist_token = tf.Variable(tf.random.uniform([1, 50]), trainable=True)
print(mnist_token)

token = tf.keras.layers.Input(shape=[50])
data = tf.keras.layers.Input(shape=[784])

weights = weight_generator(token)
weights = tf.squeeze(weights)
split_weights = tf.split(weights, weights_sizes, axis=0)
split_weights = [tf.reshape(weight, shape) for weight, shape in
                 zip(split_weights, weights_shapes)]

outputs = inner_model(split_weights, data)
joint_model = tf.keras.Model(inputs=[token, data], outputs=outputs)

# %%

rnd_data = tf.random.uniform([10, 784])
test_output = joint_model([mnist_token, rnd_data])

# %%

ds = tfds.load(name='mnist', as_supervised=True)

def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, shape=[-1])
    x /= 255
    return x, y

ds['train'] = ds['train'].repeat().shuffle(20000).map(preprocess).batch(128)
ds['test'] = ds['test'].map(preprocess).batch(128)

# %%

loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
joint_model.compile(optimizer, loss_fn, metrics=['accuracy'])

# %%

# Custom training loop is important when doing more complicated stuff
# in our case the complicated thing is updating the token for a task!
# because mnist_token is trainable and we want to update it every iteration

@tf.function
def train_step(token, x, y):
    with tf.GradientTape() as tape:
        outs = joint_model([token, x])
        loss = loss_fn(y, outs)
    
    trainables = [*joint_model.trainable_weights, token]
    grads = tape.gradient(loss, trainables)
    optimizer.apply_gradients(zip(grads, trainables))
    return outs

@tf.function
def test_step(token, x, y):
    outs = joint_model([token, x])
    return outs


def train_epoch(token, ds, steps_per_epoch):
    Accu = tf.metrics.SparseCategoricalAccuracy()
    Loss = tf.metrics.SparseCategoricalCrossentropy()
    
    for x, y in ds['train'].take(steps_per_epoch):
        outs = train_step(token, x, y)
        Accu.update_state(y, outs)
        Loss.update_state(y, outs)
    print(f'TRAIN: accuracy {Accu.result():6.3f}, loss {Loss.result():6.3f}')
    Accu.reset_states()
    Loss.reset_states()
    
    for x, y in ds['test']:
        outs = test_step(token, x, y)
        Accu.update_state(y, outs)
        Loss.update_state(y, outs)
    print(f'VALID: accuracy {Accu.result():6.3f}, loss {Loss.result():6.3f}')
    Accu.reset_states()
    Loss.reset_states()


# %%

epochs = 10
for epoch in range(1, epochs+1):
    print(f'EPOCH {epoch}/{epochs}')
    train_epoch(mnist_token, ds, 2000)


# %%

print(mnist_token)

# %%
