# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import os
import numpy as np


def apply_cosine_to_angle_features(x):
    """
    Adds the cosine of the angle features
    :param x: Features matrix
    :return: Modified features matrix
    """
    columns_mins = x.min(axis=0)
    columns_maxs = x.max(axis=0)
    min_inds = np.where(columns_mins == -3.142000)
    max_inds = np.where(columns_maxs == 3.142000)
    col_indices = np.intersect1d(min_inds, max_inds)

    x[:, col_indices] = np.cos(x[:, col_indices])
    return np.c_[x,  np.cos(x[:, col_indices])]


def save_weights_to_npy_files(data_path, weights_list):
    '''
    :param data_path: Path to save the weights
    :param weights_list: Weights to save
    :return: None
    '''
    jet_counter = 0
    yes_mmc = True

    for i in range (len(weights_list)):
        w = weights_list[i]
        fn = ''
        if yes_mmc:
            fn = 'w_' + str(jet_counter) + '_jet_MMC.npy'
            yes_mmc = False
        else:
            fn = 'w_' + str(jet_counter) + '_jet_no_MMC.npy'
            yes_mmc = True

        np.save(os.path.join(data_path, fn), w)
        if i % 2 == 1:
            jet_counter += 1


def load_csv_data(data_path, sub_sample=False):
    '''
    Loads tx and returns y (class labels), tX (features) and ids (event ids)
    :param data_path:
    :param sub_sample:
    :return: class labels, features and event ids
    '''
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    x[:, 1] = yb

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids, x


def predict_labels(w, tx):
    '''
    Generates class predictions given w, and a test tx matrix
    :param w: vector of weigths
    :param tx: matrix of features
    :return: predictions
    '''
    y_pred = tx @ w
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred


def create_csv_submission(ids, y_pred, name):
    '''
    change
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    :param ids: event ids
    :param y_pred: predictions
    :param name: name of the submission
    :return: None
    '''
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    '''
    Generate a minibatch iterator for a x.
    Takes as input two iterables (here the output desired values 'y' and the input tx 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original tx messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    :param y: class labels
    :param tx: matrix of features
    :param batch_size: number of samples per batch
    :param num_batches:
    :param shuffle: Boolean indicating if the
    :return: Desired number of batches
    '''
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def split_data(x, y, ratio, seed=1):
    '''
    split the x based on the split ratio. If ratio is 0.8
    you will have 80% of your tx set dedicated to training
    and the rest dedicated to testing
    :param x: matrix of features
    :param y: class labels
    :param ratio: |train| / (|train| + |test|)
    :param seed: random seed to sample the data
    :return: Splitted matrix features and labels
    '''
    # set seed
    np.random.seed(seed)

    N = x.shape[0]
    indices = np.random.permutation(N)
    train_stop = int(ratio * N)
    train_indices = indices[: train_stop]
    test_indices = indices[train_stop:]
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]
    return x_train, y_train, x_test, y_test


def compute_mse_loss(y, tx, w):
    '''
    Computes the MSE loss
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param w: weights vector
    :return: MSE loss
    '''
    N = tx.shape[0]
    e = y - tx @ w
    return np.squeeze(1 / (2 * N) * e.T @ e)


def compute_mse_gradient(y, tx, w):
    '''

    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param w: weights vector
    :return: gradient of the mse loss
    '''
    N = tx.shape[0]
    e = y - tx @ w
    grad = - (tx.T @ e) / N
    return grad


def least_square_one_step_GD(y, tx, w, gamma):
    '''

    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param w:  weights vector
    :param gamma:
    :return: current loss and weight vector
    '''
    loss = compute_mse_loss(y, tx, w)
    grad = compute_mse_gradient(y, tx, w)
    w = w - gamma * grad
    return loss, w


def sigmoid(t):
    '''
    Applies the sigmoid function. Slight modification for stabulity reasons
    :param t: input
    :return: sigmoid applied to input
    '''
    return 1. / (1 + np.exp(-t))


def compute_log_likelihood_loss(y, tx, w):
    '''
    Computes the log-likelihood loss
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param w: weights vector
    :return: log-likelihood loss
    '''
    y_hat = sigmoid(tx @ w)
    loss = -y.T @ np.log(y_hat) - (1 - y).T @ np.log(1 - y_hat)
    return np.squeeze(loss)


def compute_log_likelihood_gradient(y, tx, w):
    '''
    Compute the gradient of the log-likelihood loss
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param w: weights vector
    :return: gradient of the loss
    '''
    y_hat = sigmoid(tx @ w)
    e = y_hat - y
    grad = tx.T @ e
    return grad


def logistic_regression_one_step_GD(y, tx, w, gamma):
    '''
    Does one step of gradient descent on the logistic regression loss
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param w: weights vector
    :param gamma:
    :return: current loss and weight vector
    '''
    loss = compute_log_likelihood_loss(y, tx, w)
    grad = compute_log_likelihood_gradient(y, tx, w)
    w = w - gamma * grad
    return loss, w


def logistic_regression_one_step_SGD(y, tx, w, gamma):
    '''
    Does one step of stochastic gradient descent on the logistic regression loss
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param w: weights vector
    :param gamma:
    :return: current loss and weight vector
    '''
    N = tx.shape[0]
    batch_size = 16
    num_batches = (N // batch_size) // 8
    for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
        # Get current gradient
        grad = compute_log_likelihood_gradient(y_batch, tx_batch, w)

        # Update parameters
        w = w - gamma * grad
    loss = compute_log_likelihood_loss(y, tx, w)
    return loss, w


def reg_logistic_regression_one_step(y, tx, w, gamma, lambda_):
    '''
    Does one step of gradient descent on the regularized logistic regression loss
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param w: weights vector
    :param gamma: learning rate
    :param lambda_: hyperparameter for the L2 regularization
    :return: current loss and weight vector
    '''
    loss = compute_log_likelihood_loss(y, tx, w)
    loss = loss + lambda_ * (w.T @ w)
    grad = compute_log_likelihood_gradient(y, tx, w)
    grad += 2 * lambda_ * w
    w = w - gamma * grad
    return loss, w


def standardize(x):
    '''
    Standardize the original x set
    :param x: Matrix of features
    :return: Standardize features, per-column mean, per-column std
    '''
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def build_k_indices(N, k_fold, seed):
    '''
    build k indices for k-fold
    :param N: number of samples
    :param k_fold: number of fold for the cross-validation
    :param seed: random seed to sample the data
    :return: vector of indices
    '''
    interval = int(N / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(N)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def build_k_splits(N, k_fold, seed):
    '''
    Prepares the data for cross-validation
    :param N: number of samples
    :param k_fold: number of fold for the cross-validation
    :param seed: random seed to sample the data
    :return: Partitioned samples (train/test)
    '''
    k_indices = build_k_indices(N, k_fold, seed)
    train_partitions = {}
    test_partitions = {}
    for k in range(k_indices.shape[0]):
        test_indices = k_indices[k]
        train_indices = np.copy(k_indices)
        train_indices = np.delete(train_indices, k, axis=0).flatten()
        test_partitions.update({k: test_indices})
        train_partitions.update({k: train_indices})
    return train_partitions, test_partitions


def predict_labels_logistic(tx, w):
    '''
    Predict the labels for the logistic regression method
    :param tx: an NxD feature matrix
    :param w: weights vector
    :return: predicted labels
    '''
    y_hat = sigmoid(tx @ w)
    prediction = np.round(y_hat)
    prediction[prediction == 0] = -1
    return prediction


def compute_accuracy_logistic(y, tx, w):
    '''
    Computes the accuracy for the logistic regression
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param w: weights vector
    :return: acuracy and the number of properly classified samples
    '''
    N = y.shape[0]
    y_hat = sigmoid(tx @ w)
    prediction = np.round(y_hat)
    nb_corrects = np.sum(np.equal(np.squeeze(y), np.squeeze(prediction)))
    accuracy = nb_corrects / N
    return accuracy, nb_corrects


def compute_accuracy(y, tx, w):
    '''
    Computes the accuracy for every methods but logistic regression
    :param y: training labels vector
    :param tx: an NxD feature matrix
    :param w: weights vector
    :return: Accuracy and the number of properly classified samples
    '''
    N = y.shape[0]
    prediction = predict_labels(w, tx)
    nb_corrects = np.sum(np.equal(np.squeeze(y), np.squeeze(prediction)))
    accuracy = nb_corrects / N
    return accuracy, nb_corrects


def build_poly(x, degree, fractional=False, needs_first_order=True):
    '''
    Computes the polynomial expansion of a given feature
    :param x: Single feature to be polynomially expanded
    :param degree: Maximum degree for the polynomial expansion
    :param fractional: Boolean indicating if the degree are fractional or not
    :param needs_first_order: Boolean indicating if the original features should be included in the results
    :return: Matrix of the polynomially expanded feature
    '''
    N = x.shape[0]
    if needs_first_order:
        phi = np.zeros((N, degree))
        for j in range(1, degree + 1):
            if fractional:
                phi[:, j-1] = np.sign(x) * np.power(np.abs(x), 1 / j)
            else:
                phi[:, j - 1] = np.power(x, j)
    else:
        phi = np.zeros((N, degree-1))
        for j in range(2, degree + 1):
            if fractional:
                phi[:, j-2] = np.sign(x) * np.power(np.abs(x), 1 / j)
            else:
                phi[:, j - 2] = np.power(x, j)
    return phi


def expand_degrees(x, max_degree, fractional=False, needs_first_order=True):
    '''
    Coomputes the polynomially expanded features matrix
    :param x: Matrix of features
    :param max_degree: Maximum degree for the polynomial expansion
    :param fractional: Boolean indicating if the degree are fractional or not
    :param needs_first_order: Boolean indicating if the original features should be included in the results
    :return: Matrix of polynomially expanded features
    '''
    N = x.shape[0]
    D = x.shape[1]
    if needs_first_order:
        phi = np.zeros((N, max_degree * D))
        for col in range(D):
            phi[:, max_degree * col: max_degree * (col + 1)] = build_poly(x[:, col], max_degree, fractional,
                                                                                      needs_first_order)
    else:
        phi = np.zeros((N, (max_degree - 1) * D))
        for col in range(D):
            phi[:, (max_degree - 1) * col: (max_degree - 1) * (col + 1)] = build_poly(x[:, col], max_degree, fractional,
                                                                                      needs_first_order)
    return phi


def logistic_regression_plot(train_y, train_tx, test_y, test_tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD
    :param test_tx: Test features
    :param test_y: Test labels
    :param train_tx: Train features
    :param train_y: Train labels
    :param initial_w: Initial weights
    :param max_iters: Maximum number of iterations
    :param gamma: Learning rate
    :return: Current loss, current weight, log of train and test accuracies
    """
    train_accuracies = []
    test_accuracies = []
    w = initial_w
    for iter in range(max_iters):
        loss, w = logistic_regression_one_step_GD(train_y, train_tx, w, gamma)
        if iter % 10 == 0:
            train_accuracy = compute_accuracy_logistic(train_y, train_tx, w)
            test_accuracy = compute_accuracy_logistic(test_y, test_tx, w)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
    return loss, w, train_accuracies, test_accuracies


def select_feature_basis(x, threshold):
    '''
    Select a subset of features based on the Spearson correlation coefficient
    :param x: Features matrix
    :param threshold: Maximum correlation allowed in the feature matrix
    :return: Subset of little correlated features
    '''
    corr_tilde = np.corrcoef(x.T) - np.identity(x.shape[1])
    correlated_features = np.where(np.abs(corr_tilde) > threshold)
    single_pairs = []

    for feature1, feature2 in zip(correlated_features[0].tolist(), correlated_features[1].tolist()):
        if (feature1, feature2) not in single_pairs and (feature2, feature1) not in single_pairs:
            single_pairs.append((feature1, feature2))

    features_to_keep = np.arange(0, x.shape[1]).tolist()
    for feature1, feature2 in single_pairs:
        if feature2 in features_to_keep:
            features_to_keep.remove(feature2)
    return x[:, features_to_keep], features_to_keep
