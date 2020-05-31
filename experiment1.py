import random

import tensorflow as tf

from datasets import mnist
from models import hnet_with_chunks
from training_utils import get_training_function

# TODO: THESE ARE CONSTANTS TO BE READ FROM experiments.yaml:
RUN_NAME = 'multitask_testing'
LOGDIR = 'logs'
BATCH_SIZE = 128
EMBEDDING_DIM = 50
N_CHUNKS = 40
HNET_HIDDEN_DIMS = [25]
INNER_NET_DIMS = [784, 300, 10]
L2REG = 1e-4

task_generator = mnist.PermutedMNIST(bs=BATCH_SIZE)
model = hnet_with_chunks.create(embedding_dim=EMBEDDING_DIM,
                                n_chunks=N_CHUNKS,
                                hnet_hidden_dims=HNET_HIDDEN_DIMS,
                                inner_net_dims=INNER_NET_DIMS,
                                l2reg=L2REG)

rnd_idx = random.randint(1000, 9999)
writer = tf.summary.create_file_writer(f'{LOGDIR}/{RUN_NAME}{rnd_idx}')


def train_on_task(model, task_idx, iterations):
    """
    Trains the model in PermutedMNIST subtask and generates the token for task
    :param model: `tf.keras.Model` instance
    :param task_idx: an integer, for Permuted MNIST from 0 to 99
    :param iterations: number of batches to train the task from
    :return: learned embedding of the task with index `task_idx`
    """
    task_embedding = tf.random.normal([EMBEDDING_DIM]) / 10
    task_embedding = tf.Variable(task_embedding, trainable=True)

    train_ds = task_generator.get_iterator(split='train', task=task_idx)
    test_ds = task_generator.get_iterator(split='test', task=task_idx)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    train_epoch = get_training_function(model, loss_fn, optimizer)
    train_epoch(token=task_embedding,
                steps_per_epoch=iterations,
                train_ds=train_ds,
                test_ds=test_ds)
    return task_embedding


def test_on_task(model, task_idx, task_embedding):
    """
    :param model: `tf.keras.Model` instance
    :param task_idx: an integer, for Permuted MNIST from 0 to 99
    :param task_embedding: float vector of embeddings for task, with dimensionality
        100 for first lenet
    :return: accuracy float between 0.0 and 1.0
    """
    test_ds = task_generator.get_iterator(split='test', task=task_idx)
    accuracy = tf.metrics.SparseCategoricalAccuracy()

    for x, y in test_ds:
        outs, _ = model([task_embedding, x], training=False)
        accuracy.update_state(y, outs)
    return accuracy.result()


task_data = {}

for task_idx in range(10):
    print(f'LEARNING TASK {task_idx}')
    embed = train_on_task(model,
                          task_idx=task_idx,
                          iterations=4000)
    _, inner_net_weights = model([embed, tf.random.uniform([1, 784])])
    task_data[task_idx] = (embed, inner_net_weights)

    acc_on_0 = test_on_task(model, task_idx=0, task_embedding=task_data[0][0])
    with writer.as_default():
        tf.summary.scalar(name='task0_acc', data=acc_on_0, step=task_idx)

    # TODO: add old task regularization to stop catastrophic forgetting
