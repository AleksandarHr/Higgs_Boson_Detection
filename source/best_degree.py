from helpers import *
from implementations import *
from data_preprocessing import *
import os
import matplotlib.pyplot as plt


def search_best_degree(train_subsets, max_degrees, fractional=False):
    """
    A function which plots accuracy vs degree for polynomial expansion for the subsets in
    the given list of data subsets
    Arguments: train_subsets - an array of data subsets
               max_degrees - an array of maximum degrees to try
    """
    nb_test_corrects_total = []
    nb_test_samples = 0
    for i, train_subset in enumerate(train_subsets):
        N = train_subset.shape[0]
        k_fold = 4
        train_partitions, test_partitions = build_k_splits(N, k_fold=k_fold, seed=0)
        train_y = train_subset[:, 1]
        train_y = np.expand_dims(train_y, axis=1)
        train_x = train_subset[:, 2:]
        nb_test_corrects_subset = []
        nb_test_samples += test_partitions[0].shape[0]
        for max_degree in max_degrees:
            test_corrects_k_fold = []
            for k in range(0, k_fold):
                phi = expand_degrees(train_x, max_degree, fractional=fractional)
                train_phi = phi[train_partitions[k]]
                train_y_ = train_y[train_partitions[k]]
                test_phi = phi[test_partitions[k]]
                test_y = train_y[test_partitions[k]]
                train_phi, train_mean, train_std = standardize(train_phi)
                test_phi = (test_phi - train_mean) / train_std
                train_tphi = np.c_[np.ones((train_phi.shape[0], 1)), train_phi]
                test_tphi = np.c_[np.ones((test_phi.shape[0], 1)), test_phi]
                w, _ = ridge_regression(y=train_y_, tx=train_tphi, lambda_=1e-10)
                _, test_corrects = compute_accuracy(test_y, test_tphi, w)
                test_corrects_k_fold.append(test_corrects)
            nb_test_corrects_subset.append(np.mean(np.asarray(test_corrects_k_fold)))
        nb_test_corrects_total.append(nb_test_corrects_subset)
    nb_test_corrects_total = np.asarray(nb_test_corrects_total)
    nb_test_corrects_total = np.sum(nb_test_corrects_total, axis=0) / nb_test_samples

    plot_best_degree(max_degrees, nb_test_corrects_total, fractional)

def plot_best_degree(max_degrees, test_corrects_total, fractional=False):
    if fractional:
        # Plots accuracy vs. max fractional degree
        plt.plot(max_degrees, test_corrects_total, label='Test Accuracy')
        plt.xlabel('1 / Max Degree', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Accuracy vs. Fractional Degree', fontsize=18)
        plt.legend(loc='lower right')
        plt.show()
    else:
        # Plots accuracy vs. max degree
        plt.plot(max_degrees, test_corrects_total, label='Test Accuracy')
        plt.xlabel('Max Degree', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Accuracy vs. Degree', fontsize=18)
        plt.legend(loc='lower right')
        plt.show()
