#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network
    对于一个两层的sigmoid网络的前向和反向传播

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.
    计算前向传播和交叉熵损失，对于所有参数的反向传播

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.
    注意到我们这计算的梯度和作业中不一样，它们是

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    print "X.shape", X.shape
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    print "W1.shape", W1.shape
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    h = sigmoid(X.dot(W1) + b1) #(M,H)
    print "#################"
    print "h.shape", h.shape
    print "W2.shape", W2.shape
    out = softmax(h.dot(W2) + b2) #(M,Dy)
    print "-----------------"
    cost = np.sum(-labels * np.log(out))
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    d_t = out - labels #M,Dy
    gradW2 = h.T.dot(d_t) #H,Dy
    gradH = d_t.dot(W2.T) #(M,H)
    gradW1 = X.T.dot(gradH * sigmoid_grad(h)) #X.T.dot(d_t).dot(W2.T) #Dx, H
    gradb2 = np.sum(d_t, axis = 0)
    gradb1 = np.sum(gradH * sigmoid_grad(h), axis = 0)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
