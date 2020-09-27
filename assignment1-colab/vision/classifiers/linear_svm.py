from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # инициализируем градиент нулями

    # вычислим значение функции потерь и градиент
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # delta = 1
            if margin > 0:
                loss += margin

    loss /= num_train

    # Регулярицация к функции потерь (не забудьте про нее!!!).
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Вычислите градиент функции потерь и сохраните результат в переменную dW.  #
    # В данном случае проще вычислять значение функции потерь и градиент в      #
    # функции (хотя в общем случае это не очень хорошая практика). Вам может    #
    # поднадобиться изменить код выше, чтобы вычислить градиент.                #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # инициализируем градиент нулями

    #############################################################################
    # TODO:                                                                     #
    # Реализуйте векторную версию функции потерь, сохраните результат в         #
    # переменной loss.                                                          #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Реализуйте векторую версию градиента для функции потерь и сохраните       #
    # результат в переменной dW.                                                #
    #                                                                           #
    # Подсказка: вместо того, чтобы с нуля вычислять градиент, возможно, вам    #
    # будет проще использовать промежуточные значения, которые вы получили в    #
    # процессе вычисления зачения функции потерь.                               #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
