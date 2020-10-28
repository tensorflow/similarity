# TensorFlow Similarity

tf.Similarity allows you to write a simple Keras model to embed a single item, and transforms the model into a Triplet Loss (e.g. FaceNet) or Quadruplet Loss model. It also support "multi-headed" training through the concept of "auxillary tasks".  It is aimed to make writing Similarity models easy (it's easy to write a Triplet Loss model - but it's also hard to write Triplet Loss model correctly, and even harder to debug it when it it's broken).

Triplet Loss is can be hard to get right, and there can be a lot of bumps in the road if you implement it from scratch. We aim to get the infrastructure right, so the user can focus more on their models and data, and less on the infrastructure for learning.

## Getting Started

0. Install TensorFlow 2.x: `pip install tensorflow`
1. Install TensorFlow Similarity: `pip install tensorflow-similarity`
2. Head to our [hello world notebook](https://colab.sandbox.google.com/drive/1HRK4zLSAzGHwoM6dz2A1ygHSeVQ3VHdI#scrollTo=ST8JbEUrldut) that will shows you how to use TensorFlow Similarity on MNIST dataset.

## Tutorials

Head over to `tensorflow-similarity/experiments` to find tutorials on using TensorFlow Similarity on Cifar100, mnist, fashion_mnist, and iris datasets.

## Terminology

Similarity Model - A Similarity model is trained by comparing tuples of input examples, instead of trying to predict classes for single input examples.

Tower Model or Embedding Model - A Tower (or Embedding) model is a user defined model which takes one example as input, and produces an embedding vector.

Tower - Within a Similarity model, there are multiple "Towers" consisting of a single input example, Embedding model, and output embedding.  These towers then feed into some Main Task specific logic

## Disclaimer

This is not an official Google product.
