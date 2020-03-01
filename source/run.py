from data_preprocessing import *
from correlated_features import *


def train(train_subsets, lambda_, max_degree, max_fractional_degree):
    '''
    A function to perform features engineering (polynomial expansion and
    fractional polynomial expansion) and use ridge regression for model
    learning
    :param train_subsets: a list of the 8 training subsets
    :param lambda_: the hyperparameter lambda
    :param max_degree: the maximum degree for polynomial expansion
    :param max_fractional_degree: the maximum denominator for the degree for
                                    fractional polynomial expansion
    :return: weights = a list of weights vector for each of the subsets
             train_means = a list of means for each of the subsets
             train_stds = a list of standard deviations for each of the subsets
    '''
    weights = []
    train_means = []
    train_stds = []
    for train_subset in train_subsets:
        train_y = train_subset[:, 1]
        train_y = np.expand_dims(train_y, axis=1)
        train_x = train_subset[:, 2:]
        train_x_deg = expand_degrees(train_x, max_degree=max_degree)
        train_x_fractional_deg = expand_degrees(train_x, max_degree=max_fractional_degree, fractional=True,
                                                needs_first_order=False)
        train_x = np.c_[train_x_deg, train_x_fractional_deg]
        train_x, train_mean, train_std = standardize(train_x)
        train_means.append(train_mean)
        train_stds.append(train_std)
        train_tx = np.c_[np.ones((train_x.shape[0], 1)), train_x]
        w, _ = ridge_regression(y=train_y, tx=train_tx, lambda_=lambda_)
        weights.append(w)
    return weights, train_means, train_stds


if __name__ == '__main__':
    data_path = '../Data/'
    train_name = 'train.csv'
    test_name = 'test.csv'

    # Load the raw data
    if os.path.exists(os.path.join(data_path, 'train.npy')) and os.path.exists(os.path.join(data_path, 'test.npy')):
        train_X = np.load(os.path.join(data_path, 'train.npy'))
        test_X = np.load(os.path.join(data_path, 'test.npy'))
    else:
        _, _, _, train_X = load_csv_data(os.path.join(data_path, train_name), sub_sample=False)
        _, _, _, test_X = load_csv_data(os.path.join(data_path, test_name), sub_sample=False)

    # Get the feature names
    with open(os.path.join(data_path, train_name), newline='') as f:
        reader = csv.reader(f)
        features = next(reader)
    feature_dict = {}
    [feature_dict.update({feature: i}) for i, feature in enumerate(features)]

    # Get outliers value
    outliers_value = np.min(train_X)

    # Replace the outliers with nan
    train_X[train_X == outliers_value] = np.nan
    test_X[test_X == outliers_value] = np.nan

    # Pre-process the data
    train_subsets = split_jet_num(train_X, train_X.shape[0], feature_dict)
    test_subsets = split_jet_num(test_X, test_X.shape[0], feature_dict)


    # Define the model's parameters
    max_degree = 10
    max_fractional_degree = 6
    lambda_ = 1e-11

    # Train the model
    weights, train_means, train_stds = train(train_subsets, lambda_, max_degree, max_fractional_degree)

    # Predict the labels with the trained weights
    y_predicted = []
    test_Id = []
    for test_subset, weight, train_mean, train_std in zip(test_subsets, weights, train_means, train_stds):
        test_x = test_subset[:, 1:]
        test_x_deg = expand_degrees(test_x, max_degree=max_degree)
        test_x_frac_deg = expand_degrees(max_degree=max_fractional_degree, x=test_x, fractional=True,
                                         needs_first_order=False)
        test_x = np.c_[test_x_deg, test_x_frac_deg]
        test_x = (test_x - train_mean) / train_std
        test_tx = np.c_[np.ones((test_x.shape[0], 1)), test_x]
        test_Id_subset = test_subset[:, 0]

        y_predicted_subset = predict_labels(weight, test_tx)
        y_predicted.append(y_predicted_subset)
        test_Id.append(test_Id_subset)
    y_predicted = np.concatenate(y_predicted).ravel()
    test_Id = np.concatenate(test_Id).ravel()

    # Save the submission to csv
    create_csv_submission(test_Id, y_predicted, '../Data/prediction.csv')
