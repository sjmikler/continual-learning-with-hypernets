from itertools import islice

import tensorflow as tf
from tqdm import tqdm
import numpy as np


def get_training_function(model, input_shape, loss_fn, optimizer,
                          memory_reg, task_tokens):
    """
    :param task_tokens:
    :param model: usually HyperNetwork `tf.keras.Model` model
    :param input_shape: list of ints
    :param loss_fn: tf.keras.losses.Loss instance
    :param optimizer: tf.keras.optimizers.Optimizer instance
    :return: function that takes task_token, data iterator and number of steps
        and trains the `model`
    :param memory_reg: float
    """

    n_tasks = len(task_tokens)
    empty_input = tf.zeros(input_shape)
    weights_snapshots = [model([old_token, empty_input])[1] for old_token in
                         task_tokens]
    # weights_snapshots = tf.stack(weights_snapshots)
    # task_tokens = tf.stack(task_tokens)

    @tf.function
    def train_step(token, x, y):

        with tf.GradientTape() as tape:
            outs, _ = model([token, x])
            loss = loss_fn(y, outs)
            if model.losses:
                loss += tf.add_n(model.losses)

        trainables = [*model.trainable_weights, token]
        grads = tape.gradient(loss, trainables)
        optimizer.apply_gradients(zip(grads, trainables))

        if n_tasks > 0 and memory_reg > 0:
            with tf.GradientTape() as tape:
                loss = tf.constant(0, dtype=tf.float32)

                # if n_tasks > 50:
                #     # For optimization purposes, it there are more than 40 tasks
                #     # I don't want to regularize all of them at each iteration
                #     # instead, we choose a subset of which to regularize
                #     samples = tf.random.uniform(shape=(30,), minval=0,
                #                                 maxval=n_tasks, dtype=tf.int32)
                #     samples = tf.unique(samples)[0]
                #     num_samples = tf.cast(len(samples), tf.float32)
                #
                #     for sample_idx in samples:
                #         old_token = tf.gather(task_tokens, sample_idx)
                #         weight_snapshot = tf.gather(weights_snapshots, sample_idx)
                #
                #         _, weights = model([old_token, empty_input])
                #         l2 = tf.reduce_sum(tf.square(weights - weight_snapshot))
                #         loss += memory_reg * l2 / num_samples
                # else:
                for old_token, weight_snapshot in zip(task_tokens,
                                                      weights_snapshots):
                    _, weights = model([old_token, empty_input])
                    l2 = tf.reduce_sum(tf.square(weights - weight_snapshot))
                    loss += memory_reg * l2 / n_tasks

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return outs

    @tf.function
    def test_step(token, x, y):
        outs, _ = model([token, x])
        return outs

    def train_epoch(token,
                    steps_per_epoch,
                    train_ds,
                    test_ds=None,
                    regularization=()):

        Accu = tf.metrics.SparseCategoricalAccuracy()
        Loss = tf.metrics.SparseCategoricalCrossentropy()

        tbar = tqdm(islice(train_ds, steps_per_epoch),
                    total=steps_per_epoch,
                    ascii=True)
        for x, y in tbar:
            outs = train_step(token, x, y)
            Accu.update_state(y, outs)
            Loss.update_state(y, outs)
        print(f'TRAIN: accuracy {Accu.result():6.3f}, loss {Loss.result():6.3f}')
        Accu.reset_states()
        Loss.reset_states()

        if not test_ds:
            return

        for x, y in test_ds:
            outs = test_step(token, x, y)
            Accu.update_state(y, outs)
            Loss.update_state(y, outs)

        print(f'VALID: accuracy {Accu.result():6.3f}, loss {Loss.result():6.3f}')
        return Accu.result()

    return train_epoch
