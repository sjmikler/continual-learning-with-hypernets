# %%

from itertools import islice

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from models.lenet import LeNet

ds = tfds.load(name='mnist', as_supervised=True)


def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, shape=[-1])
    x /= 255
    return x, y


ds['train'] = ds['train'].repeat().shuffle(20000).map(preprocess).batch(128)
train_iter = iter(ds['train'])

ds['test'] = ds['test'].map(preprocess).batch(128)

model = LeNet(input_shape=[784],
              n_classes=10,
              l2_reg=1e-5,
              layer_sizes=[400])
model.build(input_shape=[1, 784])
model.summary()

model.step_idx = tf.Variable(0, trainable=False)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
idx = np.random.randint(100, 1000)
writer = tf.summary.create_file_writer(logdir=f'logs/test{idx}')
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
optimizer = tf.keras.optimizers.Adam()

print(f'logging index is {idx}')


# %%

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        outs = model(x)
        loss = loss_fn(y, outs)
        loss += tf.add_n(model.losses)
    
    # assert isinstance(model.step_idx, tf.Variable)
    model.step_idx.assign_add(1)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return outs


def train_epoch(model,
                iterator,
                tb_writer,
                log_interval):
    Accu = tf.metrics.SparseCategoricalAccuracy()
    Loss = tf.metrics.SparseCategoricalCrossentropy()
    
    for x, y in iterator:
        outs = train_step(x, y)
        Accu.update_state(y, outs)
        Loss.update_state(y, outs)
        
        if not model.step_idx % log_interval:
            accu, loss = Accu.result(), Loss.result()
            Accu.reset_states()
            Loss.reset_states()
            
            with writer.as_default():
                tf.summary.scalar(name='train/accuracy', data=accu,
                                  step=model.step_idx.numpy())
                tf.summary.scalar(name='train/loss', data=loss,
                                  step=model.step_idx.numpy())


@tf.function
def test_step(x, y):
    with tf.GradientTape() as tape:
        outs = model(x)
        loss = loss_fn(y, outs)
    return outs


def test_epoch(model,
               iterator,
               tb_writer):
    Accu = tf.metrics.SparseCategoricalAccuracy()
    Loss = tf.metrics.SparseCategoricalCrossentropy()
    
    for x, y in iterator:
        outs = test_step(x, y)
        Accu.update_state(y, outs)
        Loss.update_state(y, outs)
    
    accu, loss = Accu.result(), Loss.result()
    with writer.as_default():
        tf.summary.scalar(name='test/accuracy', data=accu,
                          step=model.step_idx.numpy())
        tf.summary.scalar(name='test/loss', data=loss,
                          step=model.step_idx.numpy())


# %%

def train(model,
          train_iter,
          test_iter,
          tb_writer,
          epochs=1,
          iters_in_epoch=1000,
          logging_interval=100):
    for epoch_idx in range(epochs):
        tbar = tqdm(islice(train_iter, iters_in_epoch),
                    total=iters_in_epoch,
                    desc=f'epoch: {epoch_idx + 1}/{epochs}')
        
        train_epoch(model=model,
                    iterator=tbar,
                    tb_writer=tb_writer,
                    log_interval=logging_interval)
        
        test_epoch(model=model,
                   iterator=ds['test'],
                   tb_writer=tb_writer)


# %%

train(model, train_iter, ds['test'], writer, epochs=20)
