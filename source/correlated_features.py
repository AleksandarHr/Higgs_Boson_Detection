from helpers import *
from implementations import *
import os
import matplotlib.pyplot as plt


def search_best_threshold(train_subsets, thresholds):
    """
        A function which retrieves and keeps only a selected subset of features depending on
        correlation matrix between original features and provided threshold
        Arguments: train_subsets - an array of data subsets
                   thresholds - an array of correlation thresholds to try
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
        train_accuracy_thresh_subset = []
        test_accuracy_thresh_subset = []
        nb_test_samples += test_partitions[0].shape[0]
        nb_test_corrects_subset = []
        for threshold in thresholds:
            phi, _ = select_feature_basis(train_x, threshold)
            train_phi = phi[train_partitions[0]]
            train_y_ = train_y[train_partitions[0]]
            test_phi = phi[test_partitions[0]]
            test_y = train_y[test_partitions[0]]

            train_phi, train_mean, train_std = standardize(train_phi)
            test_phi = (test_phi - train_mean) / train_std
            train_tphi = np.c_[np.ones((train_phi.shape[0], 1)), train_phi]
            test_tphi = np.c_[np.ones((test_phi.shape[0], 1)), test_phi]
            w, _ = ridge_regression(y=train_y_, tx=train_tphi, lambda_=1e-10)
            train_accuracy, train_corrects = compute_accuracy(train_y_, train_tphi, w)
            test_accuracy, test_corrects = compute_accuracy(test_y, test_tphi, w)
            train_accuracy_thresh_subset.append(train_accuracy)
            test_accuracy_thresh_subset.append(test_accuracy)
            nb_test_corrects_subset.append(test_corrects)
        nb_test_corrects_total.append(nb_test_corrects_subset)
    nb_test_corrects_total = np.sum(nb_test_corrects_total, axis=0) / nb_test_samples

    # Plots accuracy vs. correlation threshold
    plt.plot(thresholds, nb_test_corrects_total, label='Test Accuracy')
    plt.xlabel('Threshold', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy vs. Threshold', fontsize=18)
    plt.legend(loc='lower right')
    plt.show()
