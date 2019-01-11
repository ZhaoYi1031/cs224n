# -*- coding: utf-8 -*-
import time

import numpy as np
import tensorflow as tf

from q1_softmax import softmax
from q1_softmax import cross_entropy_loss
from model import Model
from utils.general_utils import get_minibatches


class Config(object):
    """Holds model hyperparams and data information.
    保持模型超参数和数据信息

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    这个config类被用来保存不同的超参数和数据集数据参数。模型被传递一个Config()对象在实例化的时候。
    他们然后可以调用self.conf.blabla来得到超参数设置
    """
    n_samples = 1024
    n_features = 100
    n_classes = 5
    batch_size = 64
    n_epochs = 50
    lr = 1e-4


class SoftmaxModel(Model):
    """Implements a Softmax classifier with cross-entropy loss."""
    # 实现一个Softmax分类器，用交叉熵损失

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors.
        生成placeholder变量来表征输入张量

        These placeholders are used as inputs by the rest of the model building
        and will be fed data during training.
        这些占位符被用作输入，通过剩下的构造，并且在training的时候将会被喂数据

        Adds following nodes to the computational graph
        增加下面的节点给计算图

        input_placeholder: Input placeholder tensor of shape
                                              (batch_size, n_features), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape
                                              (batch_size, n_classes), type tf.int32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
        增加这些占位符给实例变量
        """
        ### YOUR CODE HERE
        self.input_placeholder = tf.placeholder(tf.float32, (Config.batch_size, Config.n_features))
        self.labels_placeholder = tf.placeholder(tf.float32, (Config.batch_size, Config.n_classes))
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for training the given step.
            创建feed_dict给训练时的给定的步骤

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If label_batch is None, then no labels are added to feed_dict.
        如果label_batch是None，那么没有labels被传递给feed_dict

        Hint: The keys for the feed_dict should be the placeholder
                tensors created in add_placeholders.

                对于feed_dict的key将会是在add_placeholders的tensors

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE HERE
        if labels_batch.all() != None:
            feed_dict = {self.input_placeholder: inputs_batch, self.labels_placeholder: labels_batch}
        #!!! 为什么我一开始写反了呢
        ### END YOUR CODE
        return feed_dict

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch of input
        data into a batch of predictions. In this case, the transformation is a linear layer plus a
        softmax transformation:
        一个线性层+softmax转化层

        yhat = softmax(xW + b)

        Hint: The input x will be passed in through self.input_placeholder. Each ROW of
              self.input_placeholder is a single example. This is usually best-practice for
              tensorflow code.
              输入x将被传递通过self.input_placeholder self.input_placeholder的每一行都是一个单独的例子
              这通常是更好的实践的例子
        Hint: Make sure to create tf.Variables as needed.
                当需要的时候，确保创建tf.Variables
        Hint: For this simple use-case, it's sufficient to initialize both weights W
                    and biases b with zeros.labels_batch
                对于这个简单的使用例子，既实例化权重W和偏置b通过zero.labels_batch是足够的

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
            pred: 一个张量(batch_size, n_classes)
        """
        ### YOUR CODE HERE
        b = tf.Variable(tf.to_float(tf.zeros((Config.n_classes,))))
        W = tf.Variable(tf.to_float(tf.zeros((Config.n_features, Config.n_classes))))
        print W.shape
        print self.input_placeholder.shape
        yhat = softmax(tf.matmul(self.input_placeholder, W) + b)
        pred = yhat
        ### END YOUR CODE
        return pred

    def add_loss_op(self, pred):
        """Adds cross_entropy_loss ops to the computational graph.
        # 增加交叉熵的操作给计算图

        Hint: Use the cross_entropy_loss function we defined. This should be a very
                    short function.
              使用我们之前定义的cross_entropy_loss函数
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE
        loss = cross_entropy_loss(self.labels_placeholder, pred)
        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.
        简历training操作

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See
        创建一个优化器，并且应用梯度对所有需要训练的变量。
        这个函数返回的Op是必须通过sess.run()来传递的给模型来训练的

        https://www.tensorflow.org/api_docs/python/tf/train/Optimizer

        for more information. Use the learning rate from self.config.

        Hint: Use tf.train.GradientDescentOptimizer to get an optimizer object.
                    Calling optimizer.minimize() will return a train_op object.
            使用tf.train.GradientDescentOptimizer来获得一个优化器对象
            调用optimizer.minimize来返回一个train_op对象

        Args:
            loss: Loss tensor, from cross_entropy_loss.
            loss: 一个loss的张量，从cross_entropy_loss
        Returns:
            train_op: The Op for training.
            train_op: 对于训练的Op
        """
        ### YOUR CODE HERE
        optimizer = tf.train.GradientDescentOptimizer(Config.lr).minimize(loss)
        train_op = optimizer
        ### END YOUR CODE
        return train_op

    def run_epoch(self, sess, inputs, labels):
        """Runs an epoch of training.

        Args:
            sess: tf.Session() object
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch in get_minibatches([inputs, labels], self.config.batch_size):
            n_minibatches += 1
            total_loss += self.train_on_batch(sess, input_batch, labels_batch)
        return total_loss / n_minibatches

    def fit(self, sess, inputs, labels):
        """Fit model on provided data.

        Args:
            sess: tf.Session()
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            losses: list of loss per epoch
        """
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, inputs, labels)
            duration = time.time() - start_time
            print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration)
            losses.append(average_loss)
        return losses

    def __init__(self, config):
        """Initializes the model.

        Args:
            config: A model configuration object of type Config
        """
        self.config = config
        self.build()


def test_softmax_model():
    """Train softmax model for a number of steps."""
    config = Config()

    # Generate random data to train the model on
    np.random.seed(1234)
    inputs = np.random.rand(config.n_samples, config.n_features)
    labels = np.zeros((config.n_samples, config.n_classes), dtype=np.int32)
    labels[:, 0] = 1

    # Tell TensorFlow that the model will be built into the default Graph.
    # (not required but good practice)
    with tf.Graph().as_default() as graph:
        # Build the model and add the variable initializer op
        model = SoftmaxModel(config)
        init_op = tf.global_variables_initializer()
    # Finalizing the graph causes tensorflow to raise an exception if you try to modify the graph
    # further. This is good practice because it makes explicit the distinction between building and
    # running the graph.
    graph.finalize()

    # Create a session for running ops in the graph
    with tf.Session(graph=graph) as sess:
        # Run the op to initialize the variables.
        sess.run(init_op)
        # Fit the model
        losses = model.fit(sess, inputs, labels)

    # If ops are implemented correctly, the average loss should fall close to zero
    # rapidly.
    assert losses[-1] < .5
    print "Basic (non-exhaustive) classifier tests pass"

if __name__ == "__main__":
    test_softmax_model()
