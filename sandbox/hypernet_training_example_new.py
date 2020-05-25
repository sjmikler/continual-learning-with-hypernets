# %%

import tensorflow as tf
import tensorflow_datasets as tfds
from models import hnet_with_chunks

mnist_token = tf.Variable(tf.random.uniform([50])/10, trainable=True)
print(mnist_token)

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


hnet = hnet_with_chunks.create(embedding_dim=50,
                               n_chunks=10,
                               hnet_hidden_dims=[1000, 1000],
                               inner_net_dims=(784, 10))
hnet.build([50, [1, 784]])
hnet.compile(optimizer, loss_fn, metrics=['accuracy'])

# %%

# Custom training loop is important when doing more complicated stuff
# in our case the complicated thing is updating the token for a task!
# because mnist_token is trainable and we want to update it every iteration

@tf.function
def train_step(token, x, y):
    with tf.GradientTape() as tape:
        outs = hnet([token, x])
        loss = loss_fn(y, outs)

    trainables = [*hnet.trainable_weights, token]
    grads = tape.gradient(loss, trainables)
    optimizer.apply_gradients(zip(grads, trainables))
    return outs

@tf.function
def test_step(token, x, y):
    outs = hnet([token, x])
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

epochs = 100
for epoch in range(1, epochs+1):
    print(f'EPOCH {epoch}/{epochs}')
    train_epoch(mnist_token, ds, 2000)

# %%

