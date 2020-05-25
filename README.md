# Continual Learning Overview

The intention is to run experiments using `.yaml` config file as in `experiment.yaml`.

`experiments.yaml` is copied from another project of mine, it's not ready to use here.

*Tasks:*
* Short term
    * ✔ Make hypernetworks learn **chunks** of smaller network, not all weights at once
    * ✔ Inject hypernetwork from playground to separate model, i.e. "Lenet Hypernet"
    * ✔ Reproduce hypernetwork result on separete task (over 90% accuracy)
    * Create `train.py` file compatibile with `models` and `datasets`
    
    
* PermutedMNIST-100
    * Prepare models to compare with, i.e. models learning all tasks at once
    * Create meta model, that generates models, as in [1]
    * Make it learn anything in continual learning scenario and generate tokens for tasks
    * Debug it, replicating results from [1]
    
    
* Split MNIST (read [1], [2]) 
    * Create dataset
    
    
* Split CIFAR-10
    * Prepare ResNet model and make it learn CIFAR-10
    
    
**This repository reproduces or uses ideas from following papers:**
* [1] Continual learning with hypernetworks, https://arxiv.org/abs/1906.00695
* [2] Three scenarios for continual learning, https://arxiv.org/pdf/1904.07734.pdf
* [3] HyperNetworks, https://arxiv.org/pdf/1609.09106.pdf
