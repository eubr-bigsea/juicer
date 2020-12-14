from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import OneHotEncoderOperation
import pytest
import pandas as pd
import numpy as np


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


def hotencoder(df, attr, alias='onehotenc_1'):
    """
    Creates the cipher to encode checking each value on each column/row
    Uses sets so values don't repeat, sort to alphabetic/numeric order
    And finally encode using cipher.
    """
    df_oper = df[attr]
    idx_backup = df_oper.index
    cipher = {col: sorted({df_oper.loc[idx, col] for idx in
                           df_oper.index}) for col in df_oper.columns}
    result = [
        [1.0 if val == df_oper.loc[idx, col] else 0.0 for col in
         df_oper.columns for val in cipher[col]] for idx in df_oper.index
    ]
    result = pd.DataFrame({alias: result})
    result.index = list(idx_backup)
    return result


# OneHotEncoder
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_one_hot_encoder_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = OneHotEncoderOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].loc[:, ['onehotenc_1']].equals(
        hotencoder(test_df, ['sepalwidth', 'petalwidth']))


def test_one_hot_encoder_big_size_dataframe_success():
    df = util.iris(['sepalwidth',
                    'petalwidth',
                    'sepallength',
                    'petallength',
                    'class'], size=150)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth', 'sepallength',
                                      'petallength', 'class'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = OneHotEncoderOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].loc[:, ['onehotenc_1']].equals(
        hotencoder(test_df, ['sepalwidth', 'petalwidth', 'sepallength',
                             'petallength', 'class']))


def test_one_hot_encoder_strings_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    df.iloc[0:4, 0] = 'cat'
    df.iloc[4, 0] = 'dog'
    df.iloc[5, 0] = 'cat'
    df.iloc[6:10, 0] = 'mouse'
    df.iloc[0:4, 1] = 'dog'
    df.iloc[4, 1] = 'cat'
    df.iloc[5, 1] = 'dog'
    df.iloc[6:10, 1] = 'bunny'
    df = df.sample(frac=1)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = OneHotEncoderOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].loc[:, ['onehotenc_1']].equals(
        hotencoder(test_df, ['sepalwidth', 'petalwidth']))


def test_one_hot_encoder_bools_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=2)
    df.iloc[0, 0] = True
    df.iloc[0, 1] = True
    df.iloc[1, 0] = False
    df.iloc[1, 1] = False
    test_df = df.copy()
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = OneHotEncoderOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].loc[:, ['onehotenc_1']].equals(
        hotencoder(test_df, ['sepalwidth', 'petalwidth']))


def test_one_hot_encoder_alias_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0},
                       'alias': 'test_result'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = OneHotEncoderOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].loc[:, ['test_result']].equals(
        hotencoder(test_df, ['sepalwidth', 'petalwidth'], 'test_result'))


def test_one_hot_encoder_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = OneHotEncoderOperation(**arguments)
    assert instance.generate_code() is None


def test_one_hot_encoder_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = OneHotEncoderOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_one_hot_encoder_missing_attributes_param_fail():
    arguments = {
        'parameters': {'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        OneHotEncoderOperation(**arguments)
    assert "Parameters 'attributes' must be informed for task" in \
           str(val_err.value)


def test_one_hot_encoder_missing_multiplicity_param_fail():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    with pytest.raises(KeyError) as key_err:
        instance = OneHotEncoderOperation(**arguments)
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "'multiplicity'" in str(key_err.value)


def test_one_hot_encoder_str_and_int_fail():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    df.iloc[1, 0] = 'cat'
    df.iloc[1, 1] = 'bunny'
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = OneHotEncoderOperation(**arguments)
    with pytest.raises(TypeError) as typ_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "argument must be a string or number" in str(typ_err.value)


def test_one_hot_encoder_nan_value_fail():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    df.iloc[1, 0] = np.NaN
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = OneHotEncoderOperation(**arguments)
    with pytest.raises(ValueError) as val_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "Input contains NaN, infinity or a value too large for" \
           " dtype('float64')" in str(val_err.value)
