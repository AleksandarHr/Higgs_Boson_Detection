from helpers import *
import numpy as np


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    '''
    Linear regression using gradient descent
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param initial_w: initial weights vector
    :param max_iters: number of maximum iterations for the gradient descent
    :param gamma: update step coefficient for the gradient descent
    :return: w = weights vector, loss = final loss
    '''
    w = initial_w
    for iter in range(max_iters):
        loss, w = least_square_one_step_GD(y, tx, w, gamma)

    loss = compute_mse_loss(y, tx, w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    '''
    Linear regression using stochastic gradient descent
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param initial_w: initial weights vector
    :param max_iters: number of maximum iterations for the stochastic gradient descent
    :param gamma: update step coefficient for the gradient descent
    :return: w = weights vector, loss = final loss
    '''
    w = initial_w
    for n in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # Get current gradient
            grad = compute_mse_gradient(y_batch, tx_batch, w)

            # Update parameters
            w = w - gamma * grad

    loss = compute_mse_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    '''
    Least squares regression using normal equations
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :return: w = weights vector, loss = final loss
    '''
    N = tx.shape[0]
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    '''
    Least squares regression using normal equations
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param lambda_: the value of the hyperparameter
    :return: w = weights vector, loss = final loss
    '''
    N = tx.shape[0]
    D = tx.shape[1]
    A = tx.T @ tx + 2 * lambda_ * N * np.identity(D)
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param initial_w: initial weights vector
    :param max_iters: number of maximum iterations for the stochastic gradient descent
    :param gamma: update step coefficient for the gradient descent
    :return: w = weights vector, loss = final loss
    """
    w = initial_w
    for iter in range(max_iters):
        loss, w = logistic_regression_one_step_GD(y, tx, w, gamma)
        if iter % 100 == 0:
            message = 'losss at iteration {} is {}'
            print(message.format(iter, loss))
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent or SGD
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param lambda_: the value for the hyperparameter
    :param initial_w: initial weights vector
    :param max_iters: number of maximum iterations for the stochastic gradient descent
    :param gamma: update step coefficient for the gradient descent
    :return: w = weights vector, loss = final loss
    """
    w = initial_w
    for iter in range(max_iters):
        loss, w = reg_logistic_regression_one_step(y, tx, w, gamma, lambda_)
    return w, loss
