#!/usr/bin/env python
# -*- coding: utf-8 -*_

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 5000

import glob
import random
import numpy as np
import os.path as op
import cPickle as pickle


def load_saved_params():
    """
    A helper function that loads previously saved parameters and resets
    iteration start.
    一个帮助的函数，加载之前保存好的参数和结果
    """
    st = 0 #是保存的最大的那个
    for f in glob.glob("saved_params_*.npy"): #之前的结果都保存在
        iter = int(op.splitext(op.basename(f))[0].split("_")[2]) # 把文件名中的后面的数字给提取出来到iter里，例如一个文件名是saved_params_5000.npy
        if (iter > st):
            st = iter

    if st > 0:
        with open("saved_params_%d.npy" % st, "r") as f:
            params = pickle.load(f) #从这个持久化的文件中读出来
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None


def save_params(iter, params):
    with open("saved_params_%d.npy" % iter, "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,
        PRINT_EVERY=10):
    # t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    """ Stochastic Gradient Descent

    随机梯度下降

    Implement the stochastic gradient descent method in this function.

    实现随机梯度下降

    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a cost and the gradient
         with respect to the arguments
    f -- 去优化的函数，它应当接受一个单独的参数、输出两个输出，是相关参数的损失和梯度
    x0 -- the initial point to start SGD from
    x0 -- SGD中初始的点
    step -- the step size for SGD 每一步SGD的步长
    iterations -- total iterations to run SGD for 总的需要执行的SGD的迭代次数
    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.

                      对于参数可能需要的后处理操作。在word2vec中，我们需要正规化单词向量以便单位长度是1

    PRINT_EVERY -- specifies how many iterations to output loss

    Return:
    x -- the parameter value after SGD finishes
    """

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000

    if useSaved: # 如果使用旧的这些文件
        start_iter, oldx, state = load_saved_params() # 获取之前的状态
        if start_iter > 0: # 如果有文件非空
            x0 = oldx # x0保存的是就的状态
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    expcost = None

    for iter in xrange(start_iter + 1, iterations + 1): # 从上次掉落的位置继续捡起来重跑
        # Don't forget to apply the postprocessing after every iteration!
        # You might want to print the progress every few iterations.

        # 不要忘了在每次迭代完之后去应用postprocessing
        # 你也许需要每几次迭代之后打印进度


        cost = None
        ### YOUR CODE HERE
        #raise NotImplementedError
        # !!!
        print "x= ", x
        cost, grad = f(x) # 来计算cost和梯度 # ??? 关于这个文件里要测试的函数的这个梯度我有点蒙
        print "cost= ", cost, " grad = ", grad
        x -= step * grad # x执行梯度下降
        postprocessing(x)
        ### END YOUR CODE

        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            print "iter %d: %f" % (iter, expcost)

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x # 最终将我们


def sanity_check():
    quad = lambda x: (np.sum(x ** 2), x * 2) # ??? 关于这个文件里要测试的函数的这个梯度我有点蒙 因为这个函数不是返回一个tuple吗：第一个参数是所有元素的和，第二个参数是乘2。这咋有梯度
    # 以及SGD是要找f(x)的最小值点是吗

    print "Running sanity checks..."
    print "rrr= ", quad(0.5)
    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    print "test 1 result:", t1
    assert abs(t1) <= 1e-6

    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
    print "test 2 result:", t2
    assert abs(t2) <= 1e-6

    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
    print "test 3 result:", t3
    assert abs(t3) <= 1e-6

    print ""


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q3_sgd.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
