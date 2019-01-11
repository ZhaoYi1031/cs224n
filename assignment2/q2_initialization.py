# -*- coding: utf-8 -*_
import numpy as np
import tensorflow as tf


def xavier_weight_init():
    """Returns function that creates random tensor.
    产生随机张量

    The specified function will take in a shape (tuple or 1-d array) and
    returns a random tensor of the specified shape drawn from the
    Xavier initialization distribution.
    这个具体的函数将把一个1-d的array作为参数，并且返回一个随机的具有特定规格的tensor（从Xavier初始化分布中获得的数据）

    Hint: You might find tf.random_uniform useful.
    你会发现tf.random_uniform 很有用
    """
    def _xavier_initializer(shape, **kwargs):
        """Defines an initializer for the Xavier distribution.
        Specifically, the output should be sampled uniformly from [-epsilon, epsilon] where
            epsilon = sqrt(6) / <sum of the sizes of shape's dimensions>
        e.g., if shape = (2, 3), epsilon = sqrt(6 / (2 + 3))

        This funct`ion will be used as a variable initializer.

        Args:
            shape: Tuple or 1-d array that species the dimensions of the requested tensor.
            一个一维的array，指定了需要的tensor的shape
        Returns:
            out: tf.Tensor of specified shape sampled from the Xavier distribution.
        """
        ### YOUR CODE HERE
        import math
        epsilon = math.sqrt(6) / tf.to_float(tf.reduce_sum(shape))
        print shape
        print epsilon
        out = tf.random_uniform(shape, minval = -epsilon, maxval = epsilon, dtype=tf.float32)
        print "out=", out
        ### END YOUR CODE
        return out
    # Returns defined initializer function.
    return _xavier_initializer


def test_initialization_basic():
    """Some simple tests for the initialization.
    """
    print "Running basic tests..."
    xavier_initializer = xavier_weight_init()
    shape = (1,)
    xavier_mat = xavier_initializer(shape)
    assert xavier_mat.get_shape() == shape

    shape = (1, 2, 3)
    xavier_mat = xavier_initializer(shape)
    assert xavier_mat.get_shape() == shape
    print "Basic (non-exhaustive) Xavier initialization tests pass"


if __name__ == "__main__":
    test_initialization_basic()
