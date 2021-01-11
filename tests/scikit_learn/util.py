"""
Utilities for testing scikit-learn usage in Lemonade.
"""
import pandas as pd
import numpy as np
import os

DATA_SETS = ['iris', 'titanic', 'wine']
DATA_DIRECTORY = 'data'


def read(name, columns=None, size=None):
    """ Reads a data set used for testing """
    data_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))), 'data')
    df = pd.read_csv(os.path.join(data_dir, name + '.csv.gz'),
                     compression='gzip')
    if columns is not None:
        df = df[columns]
    if size is not None:
        df = df[:size]
    return df


def iris(columns=None, size=None):
    return read('iris', columns, size)


def wine(columns=None, size=None):
    return read('wine', columns, size)


def titanic(columns=None, size=None):
    return read('titanic', columns, size)


def get_common_imports():
    return '\n'.join([
        'import pandas as pd', 'import numpy as np', 
        'import base64', 'import json',
        'import datetime', 'import string',
        'import functools', 'import re',
        'import hashlib',
        'global np', 'global pd', 'global base64', 
        'global json', 'global datetime', 'global string',
        'global functools', 'global re',
        'global hashlib',
    ])


def execute(code, arguments):
    final_code = '\n'.join([
        get_common_imports(),
        # 'import pdb;pdb.set_trace()',
        code
    ])
    print()
    print('=' * 10, ' testing code ', '=' * 10)
    print(final_code)

    result = {}
    exec(final_code, arguments, result)
    return result


def get_X_train_data(df, features):
    """
    Method to convert some Pandas's columns to the Sklearn input format.

    :param df: Pandas DataFrame;
    :param features: a list of columns;

    :return: a DataFrame's subset as a list of list.
    """
    column_list = []
    for feature in features:
        # Validating OneHotEncode data existence
        if isinstance(df[feature].iloc[0], list):
            column_list.append(feature)
    columns = [col for col in features if col not in column_list]

    tmp1 = df[columns].to_numpy()
    if (len(column_list) > 0):
        tmp2 = df[column_list].sum(axis=1).to_numpy().tolist()
        output = np.concatenate((tmp1, tmp2), axis=1)
    elif len(features) == 1:
        output = tmp1.flatten()
    else:
        output = tmp1
    return output.tolist()


def get_label_data(df, label):
    """
    Method to check and convert a Panda's column as a Python built-in list.

    :param df: Pandas DataFrame;
    :param labels: a list of columns;

    :return: A column as a Python list.
    """

    # Validating multiple columns on label
    if len(label) > 1:
        raise ValueError(_('Label must be a single column of dataset'))

    # Validating OneHotEncode data existence
    if isinstance(df[label[0]].iloc[0], list):
        raise ValueError(_('Label must be primitive type data'))

    y = df[label].to_numpy().tolist()
    return np.reshape(y, len(y))
