# -*- coding: utf-8 -*_
import numpy as np
import tensorflow as tf
from utils.general_utils import test_all_close


def softmax(x):
    """
    Compute the softmax function in tensorflow.

    计算tensorflow里的softmax函数

    You might find the tensorflow functions tf.exp, tf.reduce_max,
    tf.reduce_sum, tf.expand_dims useful. (Many solutions are possible, so you may
    not need to use all of these functions). Recall also that many common
    tensorflow operations are sugared (e.g. x + y does elementwise addition
    if x and y are both tensors). Make sure to implement the numerical stability
    fixes as in the previous homework!

    你也许会用到tensorflow里的函数tf.exp, tf.reduce_max, tf.reduce_sum, tf.expand_dims

    注意到在tensorflow里面一些操作是加了语法糖的，例如x+y执行的是按位乘的，如果x和y都是张量。注意和之前的作业一样去实现数值稳定性

    Args:
        x:   tf.Tensor with shape (n_samples, n_features). Note feature vectors are
                  represented by row-vectors. (For simplicity, no need to handle 1-d
                  input as in the previous homework)
    Returns:
        out: tf.Tensor with shape (n_sample, n_features). You need to construct this
                  tensor in this problem.
    """

    print "softmax"
    out = tf.exp(x) / tf.reduce_sum(tf.exp(x), axis = 1, keepdims = True)
    print out
    return out


def cross_entropy_loss(y, yhat):
    """
    Compute the cross entropy loss in tensorflow.
    在tf里计算交叉熵损失
    The loss should be summed over the current minibatch.
    应当在当前的minibatch上进行计算

    y is a one-hot tensor of shape (n_samples, n_classes) and yhat is a tensor
    of shape (n_samples, n_classes). y should be of dtype tf.int32, and yhat should
    be of dtype tf.float32.
    y是一个独热码

    The functions tf.to_float, tf.reduce_sum, and tf.log might prove useful. (Many
    solutions are possible, so you may not need to use all of these functions).
    函数tf.to_float，tf.reduce_max和tf.log可能用到
    Note: You are NOT allowed to use the tensorflow built-in cross-entropy
                functions.
    不允许使用tf内置的cross-entropy函数

    Args:
        y:    tf.Tensor with shape (n_samples, n_classes). One-hot encoded.
        y是真实值
        yhat: tf.Tensorwith shape (n_sample, n_classes). Each row encodes a
                    probability distribution and should sum to 1.
        yhat是经过softmax后的概率矩阵
    Returns:
        out:  tf.Tensor with shape (1,) (Scalar output). You need to construct this
                    tensor in the problem.
        out是tf.Tensor，形状是shape(1,)
    """

    print "softmax"
    N = y.shape[0]
    print "N=", N
    #yhat = tf.reduce_max(yhat, axis = 1, keepdims = True)
    print yhat
    out = -tf.reduce_sum(tf.to_float(y) * tf.log(yhat))# + tf.to_float((1-y)) * tf.log(1-yhat))
    print out.shape
    # ???不要出N的吗
    #out /= tf.to_float(N)
    return out

def test_softmax_basic():
    """
    Some simple tests of softmax to get you started.
    Warning: these are not exhaustive.
    """

    test1 = softmax(tf.constant(np.array([[1001, 1002], [3, 4]]), dtype=tf.float32))
    with tf.Session() as sess:
            test1 = sess.run(test1)
    test_all_close("Softmax test 1", test1, np.array([[0.26894142, 0.73105858],
                                                      [0.26894142, 0.73105858]]))

    test2 = softmax(tf.constant(np.array([[-1001, -1002]]), dtype=tf.float32))
    with tf.Session() as sess:
            test2 = sess.run(test2)
    test_all_close("Softmax test 2", test2, np.array([[0.73105858, 0.26894142]]))

    print "Basic (non-exhaustive) softmax tests pass\n"


def test_cross_entropy_loss_basic():
    """
    Some simple tests of cross_entropy_loss to get you started.
    Warning: these are not exhaustive.
    """
    y = np.array([[0, 1], [1, 0], [1, 0]])
    yhat = np.array([[.5, .5], [.5, .5], [.5, .5]])

    test1 = cross_entropy_loss(tf.constant(y, dtype=tf.int32), tf.constant(yhat, dtype=tf.float32))
    with tf.Session() as sess:
        test1 = sess.run(test1)
    expected = -3 * np.log(.5)
    test_all_close("Cross-entropy test 1", test1, expected)

    print "Basic (non-exhaustive) cross-entropy tests pass"

if __name__ == "__main__":
    test_softmax_basic()
    print "finish test_softmax_basic !!!!!"
    test_cross_entropy_loss_basic()
