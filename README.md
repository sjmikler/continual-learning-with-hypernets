# Continual Learning with Hypernetworks

*Tasks:*
* PermutedMNIST-100
    * ✔ Create meta model, that generates models, as in [1]
    * ✔ Make hypernetworks learn **chunks** of smaller network, not all weights at once
    * ✔ Inject hypernetwork from playground to separate model, i.e. "Lenet Hypernet"
    * ✔ Reproduce hypernetwork result on separate task (over 90% accuracy)
    * ✔ Make it learn anything in continual learning scenario and generate tokens for tasks
    * ✔ Create evaluations that cheats and **knows** what task to do now!
    * ✔ Test HyperNetwork in continual learning scenario
    * ✔ Use regularization to finally stop the forgetting
    * ✔ Create `train.py` to run full pipeline easily
    * Prepare models we want to compare with, models that have access to all tasks at once
    * Make sure network matches results from [1]
    * Prepare clean tensorboard logs and keep them visible in `logs` folder
    * Add plot pictures to report

  
# Continual Learning
It is learning scenario in which there are multiple distinct tasks for model to learn. The most accurate way to do this is to learn all the tasks at once. E.g. having ten different tasks of classification with ten classes each, one can sample one example from each task and construct a batch consisted of 10 examples which could be used to teach a network. This method would bring the **best possible accuracy** for the neural network. However, such learning is not considered continual learning. Sometimes, not all tasks are available at once. Sometimes, not all tasks can fit in the memory at once. These would be the cases of practical use for continual learning. Although it is usually better to train separate models for different tasks. However, we expect general artificial intelligence to be able to solve multiple tasks, because humans and animals are able to do so. This itself might be good reason to develop better continual learning models. In continual learning we want a model that doesn't forget old skills, when learning something new.

## Catastrophic forgetting
A natural thing to try when learning multiple tasks is to learn first task, do fine-tuning for rest of the tasks one after another and hope network won't forget previous tasks. Unfortunately, this is not what happens. Networks tend to forget everything they learnt after just a few iterations of learning new tasks. This is called catastrophic forgetting. Effective and simple strategy to fight it is to do rehearsal, so to perform learning steps on the replayed samples from history when learning a new task. However, it requires storing samples and becomes complicated when the number of tasks grows. When testing continual learning model, one can use so called task-oracle, which can tell what task model should predict now. There are works that avoid using task-oracle as well.

# Hypernetworks
It is a meta neural network, i.e. a network that learns to generate neural networks. It does so, by taking **task token** as input and generating a set of weights, which can be then used as kernels and biases in any neural network. In simplest case, it would be a densely connected neural network, but it could be a convolutional or recurrent neural network as well. Since for many networks there are too many weights to generate at once, we do it in chunks, as in [1]. The network must know which chunk it is generating, so all chunks have their own chunk token which is not task specific -- it is shared between all the tasks. It is important for both task token and chunk token to be learnable. The whole structure is differentiable, so we can take the derivatives with respect to tokens. The difference between those tokens is that, task token remains frozen after being its task is learnt by the network, while chunk tokens are still learnable.

## Task regularization
Task embeddings are not enough to stop catastrophic forgetting during the training. Even though such task embedding is frozen, the networks continues to modify and the same task embedding will not return the same weights for inner network after few updates. This is why we have to use **task regularization**. In short, each time when we update the network, we add a task regularization term to the loss function. This regularization is responsible for keeping generated inner networks similar to as they were when they were first learned. The consequence is, during training we need not only to keep all the task tokens, but all generated inner networks. When we finish training on task 1, we need to store the task 1 token and a snapshot of weights that hypernetwork generates when task 1 token is its input. Then when training hypernetwork on the second task, we will use regularization -- summed squared difference of previously saved snapshot and current set of weights that hypernetwork generates when task 1 token is the input. Finally, we use such regularization for each task we want hypernetwork to remember.

## Advantages of hypernetworks:
### Network compression
If the number of chunks is big, time to compute all the networks parameters might grow, since we need to run the network generator for as many times as there are chunks. From the other side, such networks has less output neurons, thus less parameters. In standard scenario with, in our case, 40 chunks, such networks is smaller than the network it generates. This can be considered as some kind of parameter sharing.

### Direct representation of tasks
For each task, we learn a separate task token (task embedding), so it reminds word vectors like in word2vec. One could check if two tasks are similar by checking the distance in the space of embeddings! For most time, we use task embeddings of size 50 which we then concatenate with chunk embedding of the same size. Since tasks embedding are much smaller than full sets of weights of a neural network, it is easy and almost free to save them for, e.g. 100 tasks. This wouldn't be the case, if we wanted to save 100 separate neural networks, each for different task. Although, in both cases the theoretical memory complexity is linear in terms of the number of tasks. In practice, in this approach storing task embeddings as good as free. When testing, we use decide on using task-oracle to get the information, which task token to use.
    
# PermutedMNIST-100 benchmark
This is a standard benchmark used in continual learning papers. It artificially creates a set of tasks using standard MNIST dataset. MNIST is a dataset of handwritten digits, there are 70000 of them split into 10 classes. The images are of size 28 by 28 and since we use dense neural networks, we reshape them to 784 dimensions each. Each new task in PermutedMNIST is created by randomly permuting all the pixels from original task in each image. Note that for densely connected neural network it does not matter, given every image in the dataset is permuted in the same way. However, brand new permutation is totally new task for the network, which is not suitable for transfer learning use. Except for global features like global pixel statistics it cannot transfer anything from one task to another. This is a basic continual learning scenario.


# Project structure

* *experiments.yaml*, available parameters:
    * **logdir**: string, name of the folder containing run logs
    * **run_name**: string, name of the folder containing run logs
    * **n_tasks_to_learn**: int($\leq 100$), number of tasks to learn from PermutedMNIST dataset
    * **batch_size**: int, network will generate gradient update from this many examples in each step
    * **task_embedding_dim**: int, each task will be represented as vector of this many dimensions. Also chunk dimensionality will be equal to this number
    * **num_chunks**: int, since deep neural networks contain too many parameters to learn at once, they will be split into this many parts
    * **hypernet_hidden_dims**: list of ints, dimensionality of dense network that generates inner neural network. This list should contain only *hidden* layer dimensionality, e.g. can be empty if the network should not contain hidden layers
    * **innernet_all_dims**: list of ints, dimensionality of dense network generated by hypernetwork. This list should contain *all* layer dimensionality, i.e. cannot be empty and must contain input and output sizes.
    * **l2_regularization**: float, parameter for vanilla l2 regularization for neural network
    * **l2_memory_regularization**: float, special regularization for continual learning with hypernetwork. To avoid catastrophic forgetting, we regularize the network and this is the regularization parameter
    * **iters_per_task**: int, hypernetwork performs this many gradient updates before beginning learning new task

    * example:
```
    logdir: logs
    run_name: 1e-2m50d40c
    n_tasks_to_learn: 100
    batch_size: 128
    task_embedding_dim: 50
    num_chuks: 40
    hypernet_hidden_dims: [25]
    innernet_all_dims: [784, 300, 10]
    l2_regularization: 1e-4
    l2_memory_regularization: 1e-2
    iters_per_task: 4000
    ‌‌ ---‌‌
    logdir: logs
    run_name: 1e-3m50d40c
    n_tasks_to_learn: 100
    batch_size: 128
    task_embedding_dim: 50
    num_chuks: 40
    hypernet_hidden_dims: [25]
    innernet_all_dims: [784, 300, 10]
    l2_regularization: 1e-4
    l2_memory_regularization: 1e-3
    iters_per_task: 4000
```

* *run_experiments.py*: use this file to run all the experiments specified in `experiments.yaml`. Use: `python run_experiments.py`
* *logs*: use this directory with tensorboard, e.g. `tensorboard --logdir logs`, then enter `http://localhost:6006` address in browser
    

**References:**
1. Continual learning with hypernetworks, <https://arxiv.org/abs/1906.00695>
2. Three scenarios for continual learning, <https://arxiv.org/abs/1904.07734>
3. HyperNetworks, <https://arxiv.org/abs/1609.09106>
