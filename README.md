# Continual Learning Overview

The intention is to run experiments using `.yaml` config file as in `experiment.yaml`.

`experiments.yaml` is copied from another project of mine, it's not ready to use here.

*Tasks:*
* PermutedMNIST-100
    * ✔ Create meta model, that generates models, as in (1)
    * ✔ Make hypernetworks learn **chunks** of smaller network, not all weights at once
    * ✔ Inject hypernetwork from playground to separate model, i.e. "Lenet Hypernet"
    * ✔ Reproduce hypernetwork result on separete task (over 90% accuracy)
    * ✔ Make it learn anything in continual learning scenario and generate tokens for tasks
    * ✔ Create evaluations that cheats and **knows** what task to do now!
    * ✔ Test HyperNetwork in continual learning scenario
    * Use regularization to finally stop the forgetting
    * Prepare models we want to compare with, models that have access to all tasks at once
    * Create `train.py` file compatibile with `models` and `datasets`
    * Create evaluations that have to infer, what task they are predicting!
    * Make sure network matches results from (1)

* Split MNIST (read (1), (2))
    * Create dataset

* Split CIFAR-10
    * Prepare ResNet model and make it learn CIFAR-10

**This repository reproduces or uses ideas from following papers:**
* (1) Continual learning with hypernetworks, <https://arxiv.org/abs/1906.00695>
* (2) Three scenarios for continual learning, <https://arxiv.org/pdf/1904.07734.pdf>
* (3) HyperNetworks, <https://arxiv.org/pdf/1609.09106.pdf>
