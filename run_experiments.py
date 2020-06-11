import random

import tensorflow as tf
import yaml

from datasets import mnist
from models import hnet_with_chunks
from training_utils import get_training_function


# %%
# DEFINE FUNCTIONS

def train_on_task(model, task_idx, iterations, memory_regularization,
                  continual_regularization_data):
    """
    Trains the model in PermutedMNIST subtask and generates the token for task
    :param model: `tf.keras.Model` instance
    :param task_idx: an integer, for Permuted MNIST from 0 to 99
    :param iterations: number of batches to train the task from
    :return: learned embedding of the task with index `task_idx`
    :param memory_regularization: float
    :param continual_regularization_data: pairs [[token1, inner_weights1],
                                                 [token2, inner_weights2],...]
    """
    embedding_dim = model.layers[0].input_shape[1] // 2
    task_embedding = tf.random.normal([embedding_dim]) / 10
    task_embedding = tf.Variable(task_embedding, trainable=True)

    train_ds = task_generator.get_iterator(split='train', task=task_idx)
    test_ds = task_generator.get_iterator(split='test', task=task_idx)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    train_epoch = get_training_function(model, loss_fn, optimizer,
                                        memory_regularization)
    train_epoch(token=task_embedding,
                steps_per_epoch=iterations,
                train_ds=train_ds,
                test_ds=test_ds,
                regularization=continual_regularization_data)
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


# %%
# LOAD CONFIG

configs = yaml.safe_load_all(open('experiments.yaml', 'r'))
for config in configs:
    print("EXPERIMENT CONFIG:")
    print(config)

    RUN_NAME = config['run_name']
    LOGDIR = config['logdir']
    l2_MEMORY = config['l2_memory_regularization']
    l2_MEMORY = float(l2_MEMORY)
    N_TASKS = config['n_tasks_to_learn']
    ITERS = config['iters_per_task']


    def get_model(conf):
        """
        :param conf: a dictionary
        :return: tf.keras.Model
        """
        model = hnet_with_chunks.create(embedding_dim=conf['task_embedding_dim'],
                                        n_chunks=conf['num_chuks'],
                                        hnet_hidden_dims=conf['hypernet_hidden_dims'],
                                        inner_net_dims=conf['innernet_all_dims'],
                                        l2reg=conf['l2_regularization'])
        return model


    model = get_model(config)
    task_generator = mnist.PermutedMNIST(bs=config['batch_size'])

    rnd_idx = random.randint(1000, 9999)
    writer = tf.summary.create_file_writer(f'{LOGDIR}/{RUN_NAME}{rnd_idx}')

    task_data = []

    for task_idx in range(N_TASKS):
        print(f'LEARNING TASK {task_idx}')
        embed = train_on_task(model,
                              task_idx=task_idx,
                              iterations=ITERS if task_idx > 0 else ITERS * 2,
                              memory_regularization=l2_MEMORY,
                              continual_regularization_data=task_data)

        empty_input = tf.zeros([1, 784])
        _, inner_net_weights = model([embed, empty_input])
        task_data.append([embed, inner_net_weights])

        with writer.as_default():
            test_task = 0
            acc = test_on_task(model,
                               task_idx=test_task,
                               task_embedding=task_data[test_task][0])

            print(f'TASK {test_task} accuracy: {acc}')
            tf.summary.scalar(name=f'task {test_task} accuracy',
                              data=acc,
                              step=task_idx)

    with writer.as_default():
        for test_task in range(N_TASKS):
            acc = test_on_task(model,
                               task_idx=test_task,
                               task_embedding=task_data[test_task][0])

            print(f'TASK {test_task} accuracy: {acc}')
            tf.summary.scalar(name=f'final tasks accuracy',
                              data=acc,
                              step=test_task)

# %%
