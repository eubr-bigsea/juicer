"""
Utilities for testing scikit-learn usage in Lemonade.
"""
import pandas as pd
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
	'import pandas as pd', 'import numpy as np'
    ])
def execute(code, arguments):
    final_code = '\n'.join([
	get_common_imports(),
	#'import pdb;pdb.set_trace()',
	code
    ])
    print()
    print('=' * 10, ' testing code ', '=' * 10)
    print(final_code)

    result = {}
    exec(final_code, arguments, result)
    return result

                          
