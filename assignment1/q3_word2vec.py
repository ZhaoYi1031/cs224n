#!/usr/bin/env python
# -*- coding: utf-8 -*_

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    实现一个正规化每一行，矩阵的每一行都有单位长度
    """

    ### YOUR CODE HERE
    x /= np.linalg.norm(x, axis = 1).reshape(x.shape[0], 1) #linalg就是第二范数(也就是平方和的平方根)了，然后axis=1按照行来
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    有两个参数，一个是要预测的词向量，另一个目标的

    对于一个预测的词向量的损失和梯度，考虑到softmax预测和交叉熵损失

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component) (3,1)
    target -- integer, the index of the target word 目标词的下标
    outputVectors -- "output" vectors (as rows) for all tokens 输出矩阵（行向量）对于所有的符号 (5, 3)
    # 我现在只想关注一下，这个变量在干嘛呢!!!
    dataset -- needed for negative sampling, unused here. 需要负采样的，在这儿没用

    Return:
    cost -- cross entropy cost for the softmax word prediction #对于词向量预测的交叉熵损失
    gradPred -- the gradient with respect to the predicted word
           vector 关于预测的词向量(vc)的梯度
    grad -- the gradient with respect to all the other word
           vectors 关于所有其它词向量(uw)的梯度

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    # predicted: (3,)
    # outputVectors: (5, 3)
    # target: scala

    D = predicted.shape[0]
    predicted = predicted.reshape(D, 1) #(3,1)
    #[[-0.59609459] [0.7795666] [0.19221644]]
    print "predicted", predicted
    out = outputVectors.dot(predicted) #把output
    # print "outputVectors", outputVectors #(5,3)
    # [[-0.6831809 - 0.04200519  0.72904007]
    #  [0.18289107  0.76098587 - 0.62245591]
    #  [-0.61517874 0.5147624 - 0.59713884]
    #  [-0.33867074 - 0.80966534 - 0.47931635]
    #  [-0.52629529 - 0.781904080.33412466]]

    correct_class_score = out[target] #[-0.35066203] #(1,)
    # print "correct_class_score", correct_class_score.shape
    exp_sum = np.sum(np.exp(out)) #(5,1)
    cost = np.log(exp_sum) - correct_class_score #我们还原一下交叉熵的现场 -log()==> exp(correct_class_score) / exp_sum

    # 下面就是计算梯度的过程了
    # 把作业题中推导出的公式带进去就OK
    margin = np.exp(out) / exp_sum
    margin[target] += -1 #
    gradPred = margin.T.dot(outputVectors) #
    grad = margin.dot(predicted.T)

    print "gradPred", gradPred.shape #(1,3)
    print "grad", grad.shape #(5,3)

    # print "target", target
    # print "outputVectors", outputVectors.shape

    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    # 为了在negativeSamplingSoftmax中采样出K个对象来更新梯度的函数
    """ Samples K indexes which are not the target """
    # 采样K个下标，不是target

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    对于word2vec模型的cost计算和梯度更新的方法，与之形成对比的是softmaxCostAndGradient。主要就是因为
    后者在计算cost的时候分母的计算太耗时。于是从

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    实现损失函数和梯度

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
    # 啊这个采样下标的部分已经为我们完成了。

    ### YOUR CODE HERE
    #raise NotImplementedError
    
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)
    
    out1 = sigmoid(outputVectors[target].dot(predicted))
    cost = -np.log(out1)
    grad[target] += (out1 - 1) * predicted
    gradPred += (out1 - 1) * outputVectors[target]

    for k in range(K):
        out2 = sigmoid(- outputVectors[indices[k+1]].dot(predicted))
        cost += -np.log(out2)
        grad[indices[k+1]] += - (out2 - 1) * predicted
        gradPred += -(out2 - 1) * outputVectors[indices[k+1]]

    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    word2vec 的 Skip-gram模型

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word 当前的中心词
    C -- integer, context size #上下文的大小
    contextWords -- list of no more than 2*C strings, the context words #不超过2*C的字符串，表示上下文单词
    tokens -- a dictionary that maps words to their indices in
              the word vector list #一个字典，映射下标到词向量
    inputVectors -- "input" word vectors (as rows) for all tokens #输入的词向量
    outputVectors -- "output" word vectors (as rows) for all tokens #对于所有token的输出的词向量

    # ???输入和输出的词向量有什么区别呢，输入的不就是只有中心词才更新了梯度了，然后输出的就是词库的一些吗

    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    print "tokens", tokens
    # tokens {'a': 0, 'c': 2, 'b': 1, 'e': 4, 'd': 3}
    print "contextWords", contextWords
    # contextWords ['c', 'a', 'b', 'b', 'b', 'b']
    print "currentWord", currentWord
    # currentWord a

    print "inputVectors", inputVectors
    print "outputVectors", outputVectors
    # inputVectors[[-0.96735714 - 0.02182641  0.25247529]
    #              [0.73663029 - 0.48088687 - 0.47552459]
    #              [-0.27323645 0.12538062 0.95374082]
    #              [-0.56713774 - 0.27178229 - 0.77748902]
    #              [-0.59609459 0.7795666 0.19221644]]
    # outputVectors[[-0.6831809 - 0.04200519  0.72904007]
    #               [0.18289107 0.76098587 - 0.62245591]
    #               [-0.61517874  0.5147624 - 0.59713884]
    #               [-0.33867074 - 0.80966534 - 0.47931635]
    #               [-0.52629529 - 0.78190408 0.33412466]]

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    source = tokens[currentWord]
    predicted = inputVectors[source] # 找到预测的词向量的

    # predicted是inputVectors里的

    for target_word in contextWords: # 对于上下文词典中的每一个目标单词
        target = tokens[target_word] # 找到id
        cost_one, gradPred, grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
        # 单个的损失， 中心词的梯度， 所有的上下文单词的梯度
        cost += cost_one
        # 损失的总和
        gradIn[source] = gradIn[source] + gradPred # 中心词的梯度更新
        gradOut += grad #

    print "cost=", cost

    print "gradIn", gradIn
    print "gradOut", gradOut

    # gradIn[ [0.          0.          0.]
    #         [0.         0.          0.]
    #         [-0.33539925 - 0.00897325 - 0.22608532]
    #         [0.             0.          0.]
    #         [0.          0.          0.]]
    # gradOut[[0.13601333 - 0.06241274 - 0.474759]
    #         [0.10879187 - 0.04992157 - 0.3797416]
    #         [-0.20314289  0.09321664   0.70907696]
    #         [0.09472154 - 0.04346509 - 0.33062865]
    #         [-0.13638384  0.06258276   0.47605228]]


    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    #raise NotImplementedError
    #!!!!
    source = [tokens[source_word] for source_word in contextWords]
    predicted = inputVectors[source] # 输入词到词向量映射
    predicted = np.sum(predicted, axis=0) #sum

    target = tokens[currentWord]
    cost_one, gradPred, grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
    cost += cost_one
    # +=自更新报错？？？
    for i in source:
        gradIn[i] = gradIn[i] + gradPred #输入上下文词向量更新
    gradOut += grad
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
# 需要检查的函数
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):

    batchsize = 50 #batchsize为50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:] # ??? 是说选取后面的一半作为输入词向量
    outputVectors = wordVectors[N/2:,:] #           前面的一半作为输出词向量吗
    print "inputVectors", inputVectors
    print "outputVectors", outputVectors

    # inputVectors[[-0.96735714 - 0.02182641  0.25247529]
    #             [0.73663029 - 0.48088687 - 0.47552459]
    #             [-0.27323645 0.12538062 0.95374082]
    #             [-0.56713774 - 0.27178229 - 0.77748902]
    #             [-0.59609459 0.7795666 0.19221644]]
    # outputVectors[[-0.6831809 - 0.04200519  0.72904007]
    #             [0.18289107 0.76098587 - 0.62245591]
    #             [-0.61517874  0.5147624 - 0.59713884]
    #             [-0.33867074 - 0.80966534 - 0.47931635]
    #             [-0.52629529 - 0.78190408 0.33412466]]

    for i in xrange(batchsize): # 跑batchsize次迭代
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1) # 中心词，上下文单词都是随机选的

        print "centerword=", centerword
        print "context=", context

        if word2vecModel == skipgram: # ??? 这下面的两个为什么都是1
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom # cost只是方便我们查看的，最终这个cost在收敛的时候要小于10才行，
        grad[:N/2, :] += gin / batchsize / denom # 梯度更新 # ??? 输入和输出的词向量为什么都要进行更新呢
        grad[N/2:, :] += gout / batchsize / denom #

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
