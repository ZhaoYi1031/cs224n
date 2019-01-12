# -*- coding: utf-8 -*-
import copy
class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.

        实例化这个部分的解析

        Your code should initialize the following fields:
            self.stack: The current stack represented as a list with the top of the stack as the
                        last element of the list.
                        当前的栈，用一个list，其中栈顶是list的最后的元素
            self.buffer: The current buffer represented as a list with the first item on the
                         buffer as the first item of the list
                        当前的buffer，最前的元素是最前的元素
            self.dependencies: The list of dependencies produced so far. Represented as a list of
                    tuples where each tuple is of the form (head, dependent).
                    Order for this list doesn't matter.
                    当前产生的依赖，用一个list其中有很多tuples，每一个tuple形式是(head, dependent)
                    这个list的顺序无所谓

        The root token should be represented with the string "ROOT"
        token的跟应当是"root"

        Args:
            sentence: The sentence to be parsed as a list of words.
                      Your code should not modify the sentence.
        """
        # The sentence being parsed is kept for bookkeeping purposes. Do not use it in your code.
        self.sentence = sentence
        # print "sentence = ", sentence

        ### YOUR CODE HERE
        self.stack = ["ROOT"]
        self.buffer = copy.copy(self.sentence)
        self.dependencies = []
        ### END YOUR CODE

    def parse_step(self, transition):
        """Performs a single parse step by applying the given transition to this partial parse
        实现一个单步的parse step通过应用给定的transition给这个partial parse

        Args:
            transition: A string that equals "S", "LA", or "RA" representing the shift, left-arc,
                        and right-arc transitions. You can assume the provided transition is a legal
                        transition.
        """
        ### YOUR CODE HERE
        # print "parse_step"
        # print "self.sentence = ", self.sentence
        # print transition
        flag = False
        for c in transition:
            if len(self.buffer) == 0 and len(self.stack) <= 1:
                break
            if flag == True:
                continue
                flag = False
            # print "c=", c, self.buffer
            if c == 'S': # SHIFT: removes the first word from the buffer and pushes it onto the stack.
                val = self.buffer[0]
                # print "self.s =", self.sentence
                self.buffer.remove(self.buffer[0])
                # print "self.sentenceeee=", self.sentence
                self.stack.append(val)
                flag = False
            elif c == 'L':
                # LEFT-ARC: marks the second (second most recently added) item on the stack as a dependent of the
                # first item and removes the second item from the stack.
                val1 = self.stack[len(self.stack) - 1]
                val2 = self.stack[len(self.stack) - 2]
                self.dependencies.append((val1, val2))
                self.stack.remove(self.stack[len(self.stack) - 2])
                # print "self.sssssssss=", self.sentence

                flag = True

            elif c == 'R':
                # RIGHT-ARC: marks the first (most recently added) item on the stack as a dependent of the second item
                # and removes the first item from the stack.
                val1 = self.stack[len(self.stack) - 1]
                val2 = self.stack[len(self.stack) - 2]
                self.dependencies.append((val2, val1))
                self.stack.remove(self.stack[len(self.stack) - 1])
                # print "self.sssss=", self.sentence

                flag = True


        # print "end parse_step"

        ### END YOUR CODE

    def parse(self, transitions):
        """Applies the provided transitions to this PartialParse

        Args:
            transitions: The list of transitions in the order they should be applied
        Returns:
            dependencies: The list of dependencies produced when parsing the sentence. Represented
                          as a list of tuples where each tuple is of the form (head, dependent)
        """
        #print "\n\n\n\n%s\n\n\n\n\n\n"%self.sentence
        # print transitions
        for transition in transitions:
            self.parse_step(transition)
        #print "#######################\n%s##############\n"%self.sentence
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.
    执行一个minibatch中的所有的句子

    Args:
        sentences: A list of sentences to be parsed (each sentence is a list of words)
                    一个要被解析的句子的list，每一个单词是一个word的list
        model: The model that makes parsing decisions. It is assumed to have a function
               model.predict(partial_parses) that takes in a list of PartialParses as input and
               returns a list of transitions predicted for each parse. That is, after calling
                   transitions = model.predict(partial_parses)
               transitions[i] will be the next transition to apply to partial_parses[i].
               这个模型执行解析的决定。它被认为有一个函数，返回一个list的tranistions准备给每一个parse。
               即，调用transition = model.predict(paritial_parses)
        batch_size: The number of PartialParses to include in each minibatch
                    在每次minibatch中执行的ParitialParses的次数
    Returns:
        dependencies: A list where each element is the dependencies list for a parsed sentence.
                      Ordering should be the same as in sentences (i.e., dependencies[i] should
                      contain the parse for sentences[i]).
                      一个list，每一个元素是要解析的句子的list
                      顺序要和sentences相同
    """

    ### YOUR CODE HERE
    N = len(sentences)
    print "N =", N
    print "batch_size=", batch_size

    # pp = PartialParse([])
    # pp.stack, pp.buffer, pp.dependencies = stack, buf, deps
    # pp.parse_step(transition)
    # stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))

    """
    j = 0
    dependencies = []
    while j < N:
        partial_parses = []
        sens = []
        for i in range(batch_size):
            pp = PartialParse(sentences[j])
            sens.append(sentences[j])
            partial_parses.append(pp)
            j = j+1

        transitions = model.predict(partial_parses)
        for i in range(batch_size):
            dependencies_one = PartialParse(sens[i]).parse(transitions)
            print "dependencies_one = ", dependencies_one
            dependencies.extend(dependencies_one)
    ### END YOUR CODE
    """
    # 我总算大概差不多知道意思是什么了，就是说NN在跑并发的时候的效率更高
    # 在这里我们就希望多个sentence的parse可以同时执行,而不是让它一个sentence一起都parse_step完事了
    partial_parses = []
    dependencies = []
    for s in sentences:
        #print "@"
        partial_parses.append(PartialParse(s))
        #print "@@"
    unfinished_parses = partial_parses[:] #切片就是浅拷贝啦（其实不切片应该就是最原始的拷贝(全是引用)了吧
    # print "unfinished_parses", unfinished_parses
    # print "~~~~~~~~~~~"
    while len(unfinished_parses) > 0:
        batch_parses = unfinished_parses if len(unfinished_parses) <= batch_size else unfinished_parses[0: batch_size]
        # print "Before model.predict"
        len_batch_parses = len(batch_parses)# print batch_parses
        transitions = model.predict(batch_parses)
        # print "After model.predict"
        delete_parses = []
        i = 0
        # print "transitions =", transitions
        while i < len(batch_parses):
            # print "i=", i
            batch_parses[i].parse_step(transitions[i])
            if len(batch_parses[i].buffer) == 0 and len(batch_parses[i].stack) == 1:
                delete_parses.append(i)
                #unfinished_parses.remove(unfinished_parses[i])
            i += 1
    	for _ in reversed(delete_parses):
            # print _, "  ", len(unfinished_parses), " ", len_batch_parses, " ", 
    		del unfinished_parses[_]

    for i in partial_parses:
        dependencies.append(i.dependencies)
    return dependencies


def test_step(name, transition, stack, buf, deps,
              ex_stack, ex_buf, ex_deps):
    """Tests that a single parse step returns the expected output"""
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps

    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
    assert stack == ex_stack, \
        "{:} test resulted in stack {:}, expected {:}".format(name, stack, ex_stack)
    assert buf == ex_buf, \
        "{:} test resulted in buffer {:}, expected {:}".format(name, buf, ex_buf)
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)
    print "{:} test passed!".format(name)


def test_parse_step():
    """Simple tests for the PartialParse.parse_step function
    Warning: these are not exhaustive
    """
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat",), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run",), (), (("run", "fast"),))


def test_parse():
    """Simple tests for the PartialParse.parse function
    Warning: these are not exhaustive
    """
    sentence = ["parse", "this", "sentence"]
    dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
    print "1 sentence===", sentence
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    assert dependencies == expected,  \
        "parse test resulted in dependencies {:}, expected {:}".format(dependencies, expected)
    print "now time sentence = !!!!", sentence
    assert tuple(sentence) == ("parse", "this", "sentence"), \
        "parse test failed: the input sentence should not be modified"
    print "parse test passed!"


class DummyModel(object):
    """Dummy model for testing the minibatch_parse function
    First shifts everything onto the stack and then does exclusively right arcs if the first word of
    the sentence is "right", "left" if otherwise.
    测试 minibatch_parse 的虚拟模型
    """
    def predict(self, partial_parses):
        # for pp in partial_parses:
            # print "pp.stack", pp.stack
            # print "pp.buffer", pp.buffer
        # 不要深究这个虚拟模型，它只是随便起的给baby test的，实际上stack不会是right 这些东西啥的
        result = [("RA" if pp.stack[1] is "right" else "LA") if len(pp.buffer) == 0 else "S"
                    for pp in partial_parses]
        # print result
        # print "\n\n\n\n\n"
        return result#[("RA" if pp.stack[1] is "right" else "LA") if len(pp.buffer) == 0 else "S"
               # for pp in partial_parses]
        # 如果pp.stack[1]是right，那么就是RA 否则就是LA


def test_dependencies(name, deps, ex_deps):
    """Tests the provided dependencies match the expected dependencies"""
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)


def test_minibatch_parse():
    """Simple tests for the minibatch_parse function
    Warning: these are not exhaustive
    """
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    print "@@@@@@@@@@@@@@@@@"
    print deps
    print "@@@@@@@@@@@@@@@@@"
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))
    print "minibatch_parse test passed!"

if __name__ == '__main__':
    test_parse_step()
    test_parse()
    test_minibatch_parse()
