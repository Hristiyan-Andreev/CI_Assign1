#!/usr/bin/env python
import numpy as np

from logreg_toolbox import sig
from logreg_toolbox import poly_2D_design_matrix

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) (Logistic Regression)
TODO Fill the cost function and the gradient
"""


def cost(theta, x, y):
    """
    Cost of the logistic regression function.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: cost
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #
    # Write the cost of logistic regression as defined in the lecture
    # Hint:
    #   - use the logistic function sig imported from the file toolbox
    #   - sums of logs of numbers close to zero might lead to numerical errors, try splitting the cost into the sum
    # over positive and negative samples to overcome the problem. If the problem remains note that low errors is not
    # necessarily a problem for gradient descent because only the gradient of the cost is used for the parameter updates.
    hypo = sig(np.dot(x,theta))

    truehypoindexes = np.where(y)[0]
    falsehypo = np.delete(hypo, truehypoindexes)

    cost0 = sum(-np.log(1 - falsehypo))
    cost1 = sum(-np.log(hypo[truehypoindexes]))

    c = (cost0+cost1)/N

    # END TODO
    ###########

    return c


def grad(theta, x, y):
    """

    Compute the gradient of the cost of logistic regression

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: gradient
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #
    hypo = sig(np.dot(x, theta))
    g = np.zeros(theta.shape)
    error = hypo-y
    g = (1/N) * (np.dot(error.T, x))
    # END TODO
    ###########

    return g
