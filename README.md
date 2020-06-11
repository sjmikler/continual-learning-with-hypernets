# Continual Learning with Hypernetworks

*Tasks:*
* PermutedMNIST-100
    * ✔ Create meta model, that generates models, as in [1]
    * ✔ Make hypernetworks learn **chunks** of smaller network, not all weights at once
    * ✔ Inject hypernetwork from playground to separate model, i.e. "Lenet Hypernet"
    * ✔ Reproduce hypernetwork result on separete task (over 90% accuracy)
    * ✔ Make it learn anything in continual learning scenario and generate tokens for tasks
    * ✔ Create evaluations that cheats and **knows** what task to do now!
    * ✔ Test HyperNetwork in continual learning scenario
    * ✔ Use regularization to finally stop the forgetting
    * ✔ Create `train.py` to run full pipeline easily
    * Prepare models we want to compare with, models that have access to all tasks at once
    * Make sure network matches results from [1]
    * Prepare clean tensorboard logs and keep them visible in `logs` folder
    * Add plot pictures to report

# Hypernetwork
It is a meta neural network, i.e. a network that learns to generate neural networks. It does so, by taking **task token** as input and generating a set of weights, which can be then used as kernels and biases in any neural network. In simplest case, it would be a densly connected neural network, but it could be a convolutional or recurrent neural network as well. Since for many networks there are too many weights to generate at once, we do it in chunks, as in [1]. The network must know which chunk it is generating, so all chunks have their own chunk token which is not task specific -- it is shared between all the tasks. It is important for both task token and chunk token to be learnable. The whole structure is differentiable, so we can take the derivatives with respect to tokens. The difference between those tokens is that, task token remains frozen after being its task is learnt by the network, while chunk tokens are still learnable.

## Advantages of hypernetworks:
### Network compression
If the number of chunks is big, time to compute all the networks parameters might grow, since we need to run the network generator for as many times as there are chunks. From the other side, such networks has less output neurons, thus less parameters. In standard scenario with, in our case, 40 chunks, such networks is smaller than the network it generates. This can be considered as some kind of parameter sharing.

### Direct representation of tasks
For each task, we learn a separate task token (task embedding), so it reminds word vectors like in word2vec. One could check if two tasks are similar by checking the distance in the space of embeddings! For most time, we use task embeddings of size 50 which we then concatenate with chunk embedding of the same size. Since tasks embedding are much smaller than full sets of weights of a neural network, it is easy and almost free to save them for, e.g. 100 tasks. This wouldn't be the case, if we wanted to save 100 separate neural networks, each for different task. Although, in both cases the theoretical complexity is linear in terms of the number of tasks.

## Task regularization
Task embeddings are not enough to stop catastrophic forgetting during the training. Even though such task embedding is frozen, the networks continues to modify and the same task embedding will not return the same weights for inner network after few updates. This is why we have to use **task regularization**. In short, each time when we update the network, we add a task regularization term to the loss function. This regularization is responsible for keeping generated inner networks similar to as they were when they were first learned. The consequence is, during training we need not only to keep all the task tokens, but all generated inner networks. When we finish training on task 1, we need to store the task 1 token and a snapshot of weights that hypernetwork generates when task 1 token is its input. Then when training hypernetwork on the second task, we will use regularization -- summed squared difference of previously saved snapshot and current set of weights that hypernetwork generates when task 1 token is the input. Finally, we use such regularization for each task we want hypernetwork to remember.
    
# PermutedMNIST-100 benchmark
This is a standard benchmark used in continual learning papers. It artificialy creates a set of tasks using standard MNIST dataset. MNIST is a dataset of handwritten digits, there are 60000 of them split into 10 classes. The images are of size 28 by 28 and since we use dense neural networks, we reshape them to 784 dimensions each. Each new task in PermutedMNIST is created by randomly permuting all the pixels from original task in each image. Note that for densly connected neural network it does not matter, given every image in the dataset is permuted in the same way. However, brand new permutation is totally new task for the network, which is not suitable for transfer learning use. Except for global features like global pixel statistics it cannot transfer anything from one task to another. This is a basic continual learning scenario.


**References:**
1. Continual learning with hypernetworks, <https://arxiv.org/abs/1906.00695>
2. Three scenarios for continual learning, <https://arxiv.org/abs/1904.07734>
3. HyperNetworks, <https://arxiv.org/abs/1609.09106>
