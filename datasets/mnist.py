import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class MNIST:
    mean = 0.13066048
    std = 0.3081078


class PermutedMNIST:
    def __init__(self, bs):
        self.bs = bs

        perms = np.load('datasets/_permutations.npy')
        self.perms = tf.constant(perms)

        self.ds = tfds.load("mnist", as_supervised=True)

        def preprocess(x, y):
            x = tf.cast(x, tf.float32)
            x = (x - MNIST.mean) / MNIST.std
            x = tf.reshape(x, [784])
            return x, y

        self.ds = {key: ds.map(preprocess) for key, ds in self.ds.items()}
        self.ds["train"] = self.ds["train"].shuffle(5000).repeat()

    def get_iterator(self, split, task):
        ds = self.ds[split]
        self.task = task

        def permute(x, y):
            if self.task is None:
                self.task = tf.random.uniform([], 0, 100, tf.int32)

            x = tf.gather(x, self.perms[self.task])
            return x, y

        ds = ds.map(permute)
        ds = ds.batch(self.bs)
        return iter(ds)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot(x):
        x = x.numpy().reshape(-1, 28, 28)
        x = x[0]
        plt.imshow(x)

    ds = PermutedMNIST(16)
    train_ds = ds.get_iterator('train', task=1)

    for x, y in train_ds:
        plot(x)
        break
