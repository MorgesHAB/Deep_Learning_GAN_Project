# Deep Learning GAN Project
KTH DD2424 Deep Learning Project

Jeremy Goumaz
Alexander Klukas
Lionel Isoz


# Description
We aim to generate fake images by using different GAN (Generative Adversarial Networks) implementations. 
The goal of the project is to learn more about GAN and to try to generate good quality images. We will begin by coding a simple GAN model and run it with the MNIST dataset. Then we will do the same with a DCGAN (Deep Convolutional Generative Adversarial Network) model. If we have the time, we will try other GAN implementations/variations and compare them. We will carry out all experiments using the MNIST dataset to have a fair reference for comparison. At the end of the project, we will use other datasets to generate more interesting images than the digits offered be tge MNIST set.

# Dataset
We will use MNIST dataset which is a commonly used dataset with 28×28 pixel grayscale images of handwritten digits (0-9). We will use this dataset to test and improve our code and to test different GAN implementations. Thus it will allow us to compare implementations and find the best way to generate images.
At the end of the project, we will select the best GAN model from our experiments and we will try to generate images using other datasets (not determined yet).

# Software package
We will use TensorFlow and Keras because a lot of papers about GANs use these libraries. It also seems that TensorFlow has a larger community than PyTorch in this field. 