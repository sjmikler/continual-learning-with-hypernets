import tensorflow as tf
from tqdm import tqdm
from itertools import islice


def get_training_function(model, loss_fn, optimizer):
    """
    :param model: usually HyperNetwork `tf.keras.Model` model
    :param loss_fn: tf.keras.losses.Loss instance
    :param optimizer: tf.keras.optimizers.Optimizer instance
    :return: function that takes task_token, data iterator and number of steps
        and trains the `model`
    """
    
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
        return outs
    
    @tf.function
    def test_step(token, x, y):
        outs, _ = model([token, x])
        return outs
    
    def train_epoch(token, steps_per_epoch, train_ds, test_ds=None):
        Accu = tf.metrics.SparseCategoricalAccuracy()
        Loss = tf.metrics.SparseCategoricalCrossentropy()
        
        tbar = tqdm(islice(train_ds, steps_per_epoch),
                    total=steps_per_epoch,
                    ascii=True)
        for x, y in tbar:
            outs = train_step(token, x, y)
            Accu.update_state(y, outs)
            Loss.update_state(y, outs)
        print(
            f'TRAIN: accuracy {Accu.result():6.3f}, loss {Loss.result():6.3f}')
        Accu.reset_states()
        Loss.reset_states()
        
        if not test_ds:
            return
        
        for x, y in test_ds:
            outs = test_step(token, x, y)
            Accu.update_state(y, outs)
            Loss.update_state(y, outs)
        print(
            f'VALID: accuracy {Accu.result():6.3f}, loss {Loss.result():6.3f}')
        Accu.reset_states()
        Loss.reset_states()
    
    return train_epoch
