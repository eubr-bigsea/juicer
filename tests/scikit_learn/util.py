"""
Utilities for testing scikit-learn usage in Lemonade.
"""
import os

import pandas as pd
import duckdb
import polars as pl

from juicer.scikit_learn.util import get_label_data, get_X_train_data
from juicer.transpiler import TranspilerUtils
from typing import List, Dict
from juicer.operation import Operation

DATA_SETS = ['iris', 'titanic', 'wine', 'tips', 'funel', 'iris2']
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


def iris(columns: List[str] = None, size: int = None) -> pd.DataFrame:
    return read('iris', columns, size)

def iris_polars(columns: List[str] = None, size: int = None) -> pl.DataFrame:
    return pandas_2_polars(read('iris', columns, size))

def iris2_polars(columns: List[str] = None, size: int = None) -> pl.DataFrame:
    return pandas_2_polars(read('iris2', columns, size))

def titanic_polars(columns: List[str] = None, size: int = None) -> pl.DataFrame:
    return pandas_2_polars(read('titanic', columns, size))

def funel_polars(columns: List[str] = None, size: int = None) -> pl.DataFrame:
    return pandas_2_polars(read('funel', columns, size))

def tips_polars(columns: List[str] = None, size: int = None) -> pl.DataFrame:
    return pandas_2_polars(read('tips', columns, size))

def wine(columns: List[str] = None, size: int = None) -> pd.DataFrame:
    return read('wine', columns, size)


def titanic(columns: List[str] = None, size: int = None) -> pd.DataFrame:
    return read('titanic', columns, size)


def get_common_imports() -> str:
    return '\n'.join([
        'import pandas as pd', 'import numpy as np',
        'import base64', 'import json',
        'import datetime', 'import string',
        'import functools', 'import re',
        'import hashlib', 'import itertools',
        'import polars as pl',
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


def get_complete_code(instance: Operation):
    code: List[str] = [""]
    code.extend(instance.transpiler_utils.imports)
    code.append('')
    code.extend(instance.transpiler_utils.custom_functions.values())
    code.append('import duckdb; duckdb_global_con = duckdb.connect()')
    code.append(instance.generate_code().lstrip())
    return '\n'.join(code)


def execute(code: str, arguments: Dict[any, any]):
    final_code = '\n'.join([
        get_common_imports(),
        # 'import pdb;pdb.set_trace()',
        code
    ])
    
    result = {}
    exec(final_code, arguments, result)
    return result

def pandas_2_polars(df: pd.DataFrame):
    return pl.from_pandas(df).lazy()

def pandas_2_duckdb(df: pd.DataFrame):
    return None

def emit_event(*args, **kwargs):
    pass
