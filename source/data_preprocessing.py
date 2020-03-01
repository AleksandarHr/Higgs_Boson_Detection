import numpy as np


def split_jet_num(x, N, feature_dict):
    '''
    A function to split the original dataset into 8 subsets according to the
    categorical variable 'PRI_jet_num' [0, 1, 2, 3] and further into subsets
    containing (or not) a value for the parameter 'Der_mass_MMC'
    :param x: the original dataset
    :param N: the size of the original dataset
    :param feature_dict: a dictionary of the dataset features
    :return: a list of the eight subsets
    '''
    # Split tx wrt to 'PRI_jet_num'
    x_0_jet = x[x[:, feature_dict['PRI_jet_num']] == 0.0]
    x_1_jet = x[x[:, feature_dict['PRI_jet_num']] == 1.0]
    x_2_jet = x[x[:, feature_dict['PRI_jet_num']] == 2.0]
    x_3_jet = x[x[:, feature_dict['PRI_jet_num']] == 3.0]

    # Sanity check
    assert x_0_jet.shape[0] + x_1_jet.shape[0] + x_2_jet.shape[0] + x_3_jet.shape[0] == N

    # Split x wrt 'DER_mass_MMC'
    x_0_jet_no_MMC = x_0_jet[np.isnan(x_0_jet[:, feature_dict['DER_mass_MMC']])]
    x_0_jet_MMC = x_0_jet[np.logical_not(np.isnan(x_0_jet[:, feature_dict['DER_mass_MMC']]))]

    x_1_jet_no_MMC = x_1_jet[np.isnan(x_1_jet[:, feature_dict['DER_mass_MMC']])]
    x_1_jet_MMC = x_1_jet[np.logical_not(np.isnan(x_1_jet[:, feature_dict['DER_mass_MMC']]))]
    x_1_jet_no_MMC = np.delete(x_1_jet_no_MMC, feature_dict['DER_mass_MMC'], axis=1)

    x_2_jet_no_MMC = x_2_jet[np.isnan(x_2_jet[:, feature_dict['DER_mass_MMC']])]
    x_2_jet_MMC = x_2_jet[np.logical_not(np.isnan(x_2_jet[:, feature_dict['DER_mass_MMC']]))]
    x_2_jet_no_MMC = np.delete(x_2_jet_no_MMC, feature_dict['DER_mass_MMC'], axis=1)

    x_3_jet_no_MMC = x_3_jet[np.isnan(x_3_jet[:, feature_dict['DER_mass_MMC']])]
    x_3_jet_MMC = x_3_jet[np.logical_not(np.isnan(x_3_jet[:, feature_dict['DER_mass_MMC']]))]
    x_3_jet_no_MMC = np.delete(x_3_jet_no_MMC, feature_dict['DER_mass_MMC'], axis=1)

    # Delete columns containing nan values
    x_0_jet_no_MMC = x_0_jet_no_MMC.T[np.logical_not(np.isnan(x_0_jet_no_MMC).any(axis=0))].T
    x_1_jet_no_MMC = x_1_jet_no_MMC.T[np.logical_not(np.isnan(x_1_jet_no_MMC).any(axis=0))].T
    x_2_jet_no_MMC = x_2_jet_no_MMC.T[np.logical_not(np.isnan(x_2_jet_no_MMC).any(axis=0))].T
    x_3_jet_no_MMC = x_3_jet_no_MMC.T[np.logical_not(np.isnan(x_3_jet_no_MMC).any(axis=0))].T

    x_0_jet_MMC = x_0_jet_MMC.T[np.logical_not(np.isnan(x_0_jet_MMC).any(axis=0))].T
    x_1_jet_MMC = x_1_jet_MMC.T[np.logical_not(np.isnan(x_1_jet_MMC).any(axis=0))].T
    x_2_jet_MMC = x_2_jet_MMC.T[np.logical_not(np.isnan(x_2_jet_MMC).any(axis=0))].T
    x_3_jet_MMC = x_3_jet_MMC.T[np.logical_not(np.isnan(x_3_jet_MMC).any(axis=0))].T

    # Delete columns with std=0
    x_0_jet_no_MMC = x_0_jet_no_MMC.T[np.std(x_0_jet_no_MMC, axis=0) != 0].T
    x_1_jet_no_MMC = x_1_jet_no_MMC.T[np.std(x_1_jet_no_MMC, axis=0) != 0].T
    x_2_jet_no_MMC = x_2_jet_no_MMC.T[np.std(x_2_jet_no_MMC, axis=0) != 0].T
    x_3_jet_no_MMC = x_3_jet_no_MMC.T[np.std(x_3_jet_no_MMC, axis=0) != 0].T

    x_0_jet_MMC = x_0_jet_MMC.T[np.std(x_0_jet_MMC, axis=0) != 0].T
    x_1_jet_MMC = x_1_jet_MMC.T[np.std(x_1_jet_MMC, axis=0) != 0].T
    x_2_jet_MMC = x_2_jet_MMC.T[np.std(x_2_jet_MMC, axis=0) != 0].T
    x_3_jet_MMC = x_3_jet_MMC.T[np.std(x_3_jet_MMC, axis=0) != 0].T

    processed_data = [x_0_jet_MMC, x_1_jet_MMC, x_2_jet_MMC, x_3_jet_MMC,
                     x_0_jet_no_MMC, x_1_jet_no_MMC, x_2_jet_no_MMC, x_3_jet_no_MMC]
    return processed_data


def replace_outliers(x, replace_method):
    '''
    A function to handle the NaN values in the feature matrix either by
    replacing them with the meadian of the column or by the mean of the column
    they are in.
    :param x: the original dataset
    :param replace_method: a string for the method used to replace the
                            NaN values - 'median' or 'mean'
    :return: the IDs, the labels, and the features data
    '''
    cols_with_nan = [np.logical_or.reduce(np.isnan(col)) for col in x.T]
    cols_with_nan = np.where(cols_with_nan)[0]
    for col_with_nan in cols_with_nan:
        minus_rows = (x[:, 1] == -1)
        plus_rows = (x[:, 1] == 1)
        if replace_method == 'median':
            minus_value = np.nanmedian(x[minus_rows, col_with_nan])
            plus_value = np.nanmedian(x[plus_rows, col_with_nan])
        elif replace_method == 'mean':
            minus_value = np.nanmean(x[minus_rows, col_with_nan])
            plus_value = np.nanmean(x[plus_rows, col_with_nan])
        nan_rows = np.isnan(x[:, col_with_nan])
        x[np.logical_and(minus_rows, nan_rows), col_with_nan] = minus_value
        x[np.logical_and(plus_rows, nan_rows), col_with_nan] = plus_value
    return x[:, 0], x[:, 1], x[:, 2:]
