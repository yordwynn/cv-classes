from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Инициализируем ошибку и градиент нулями
    loss = 0.0
    dW = np.zeros_like(W)

    #################################################################################
    # TODO: Вычислите значение функции потерь softmax и ее градиент, используя      #
    # циклы. Сохраните значение потери в переменной loss и градиент в               #
    # переменной dW. Вычисления нужно выполнять аккуранто, так как в некоторых      #
    # случаях функция может вести себя нестабильно (подробнее здесь:                #
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/). #
    # Не забывайте о регуляризации.                                                 #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Инициализируем ошибку и градиент нулями
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Вычислите значение функции потерь softmax и ее градиент, не         #
    # используя циклы. Сохраните значение потери в переменной loss и градиент в #
    # переменной dW. Опять же, нужно помнить, что функция ведет себя            #
    # нестабильно в некоторых случаях. Также не забывайте про регуляризацию.    #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
