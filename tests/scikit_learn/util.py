"""
Utilities for testing scikit-learn usage in Lemonade.
"""
import os

import pandas as pd
import polars as pl

from juicer.transpiler import TranspilerUtils
from typing import List, Dict, Any
from juicer.operation import Operation
from pathlib import Path

DATA_SETS = ['iris', 'titanic', 'wine']
DATA_DIRECTORY = 'data'



def _read_data(name: str, columns: List[str] = None,
              size: int = None) -> pd.DataFrame:
    """
    Reads a data set used for testing.

    Args:
    - name (str): The name of the data set.
    - columns (list[str], optional): The columns to select from the data set.
        Defaults to None.
    - size (int, optional): The number of rows to select from the data set.
        Defaults to None.

    Returns:
    - pd.DataFrame: The read data set.
    """
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    file_path = data_dir / f"{name}.csv.gz"
    df = pd.read_csv(file_path, compression='gzip')

    if columns is not None:
        df = df[columns]

    if size is not None:
        df = df.head(size)

    return df


def iris(columns: List[str] = None, size: int = None) -> pd.DataFrame:
    return _read_data('iris', columns, size)

def iris_polars(columns: List[str] = None, size: int = None) -> pl.DataFrame:
    return pandas_2_polars(_read_data('iris', columns, size))

def iris2_polars(columns: List[str] = None, size: int = None) -> pl.DataFrame:
    return pandas_2_polars(_read_data('iris2', columns, size))

def titanic_polars(columns: List[str] = None, size: int = None) -> pl.DataFrame:
    return pandas_2_polars(_read_data('titanic', columns, size))

def funel_polars(columns: List[str] = None, size: int = None) -> pl.DataFrame:
    return pandas_2_polars(_read_data('funel', columns, size))

def tips_polars(columns: List[str] = None, size: int = None) -> pl.DataFrame:
    return pandas_2_polars(_read_data('tips', columns, size))

def wine(columns: List[str] = None, size: int = None) -> pd.DataFrame:
    return _read_data('wine', columns, size)


def titanic(columns: List[str] = None, size: int = None) -> pd.DataFrame:
    return _read_data('titanic', columns, size)


def _get_common_imports() -> str:
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
        'global hashlib', 'global itertools', 'global pl'
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
        _get_common_imports(),
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

def update_dict(d: Dict[str, Any], path: str, value: Any) -> None:
    """
    Updates a nested dictionary with a value at a specified path.

    Args:
    - d (Dict[str, Any]): The dictionary to update.
    - path (str): The path to the key to update, using dot notation
        (e.g., 'a.b.c').
    - value (Any): The value to assign to the key.

    Returns:
    - None
    """
    keys = path.split('.')
    for key in keys[:-1]:
        if '[' in key:
            k, idx = key.split('[')
            idx = int(idx[:-1])
            k = k.strip()
            if k not in d:
                d[k] = [None] * (idx + 1)
            d = d[k][idx]
        else:
            if key not in d:
                d[key] = {}
            d = d[key]
    d[keys[-1]] = value


def save_chart(data: list, layout: dict) -> None:
    """
    Saves a Plotly chart as a PNG image file.

    Args:
    - data (list): A list of Plotly graph objects.
    - layout (dict): A dictionary defining the layout of the chart.

    Returns:
    - None
    """
    import plotly.graph_objects as go

    fig = go.Figure(data=data, layout=layout)
    fig.write_image("plot.png")
