# Continual Learning Overview

The intention is to run experiments using `.yaml` config file as in `experiment.yaml`.

`experiments.yaml` is copied from another project of mine, it's not ready to use here.


*Tasks:*
* PermutedMNIST-100
    * Prepare models to compare with, i.e. models learning all tasks at once
    * Create meta model, that generates models, as in [1]
    * Make it learn anything in continual learning scenario and generate tokens for tasks
    * Debug it, replicating results from [1]
* Split CIFAR-10
    * Prepare ResNet model and make it learn CIFAR-10
    * ...
    
    

 
**This repository reproduces or uses ideas from following papers:**

[1] Continual learning with hypernetworks (https://arxiv.org/abs/1906.00695)

[2]