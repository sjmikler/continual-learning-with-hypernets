# %%

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from models import hnet_with_chunks

EMBEDDING_DIM = 100
N_CHUNKS = 40

mnist_token = tf.Variable(tf.random.uniform([EMBEDDING_DIM])/10, trainable=True)
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

hnet = hnet_with_chunks.create(embedding_dim=EMBEDDING_DIM,
                               n_chunks=N_CHUNKS,
                               hnet_hidden_dims=[50],
                               inner_net_dims=(784, 300, 10),
                               l2reg=1e-4)

hnet.build([EMBEDDING_DIM, [1, 784]])
hnet.compile(optimizer, loss_fn, metrics=['accuracy'])

# %%

# Custom training loop is important when doing more complicated stuff
# in our case the complicated thing is updating the token for a task!
# because mnist_token is trainable and we want to update it every iteration

@tf.function
def train_step(token, x, y):
    with tf.GradientTape() as tape:
        outs, _ = hnet([token, x])
        loss = loss_fn(y, outs)
        if hnet.losses:
            loss += tf.add_n(hnet.losses)

    trainables = [*hnet.trainable_weights, token]
    grads = tape.gradient(loss, trainables)
    optimizer.apply_gradients(zip(grads, trainables))
    return outs


@tf.function
def test_step(token, x, y):
    outs, _ = hnet([token, x])
    return outs


def train_epoch(token, ds, steps_per_epoch):
    Accu = tf.metrics.SparseCategoricalAccuracy()
    Loss = tf.metrics.SparseCategoricalCrossentropy()

    tbar = tqdm(ds['train'].take(steps_per_epoch),
                total=steps_per_epoch,
                ascii=True)
    for x, y in tbar:
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

epochs = 20
for epoch in range(1, epochs+1):
    print(f'EPOCH {epoch}/{epochs}')
    train_epoch(mnist_token, ds, 750)

# %%

