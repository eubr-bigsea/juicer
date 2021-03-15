"""
Utilities for testing scikit-learn usage in Lemonade.
"""
from juicer.scikit_learn.util import get_X_train_data, get_label_data
import pandas as pd
import numpy as np
import os
from juicer.transpiler import TranspilerUtils

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
        'import hashlib', 'import itertools',
        'global np', 'global pd', 'global base64', 
        'global json', 'global datetime', 'global string',
        'global functools', 'global re',
        'global hashlib', 'global itertools'
    ])


def add_minimum_ml_args(args):
    args['parameters'].update({
        'task_id': 1,
        'operation_id': 1,
        'task': {'forms': {'display_text': {'value': 0}}},
        'transpiler_utils': TranspilerUtils()
    })
    return args


def get_complete_code(instance):
    code = "\n" + \
           "\n".join(list(instance.transpiler_utils.imports)) + \
           "\n" + \
           "\n".join(instance.transpiler_utils.custom_functions.values()) + \
           "\n" + instance.generate_code().lstrip()
    return code


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

