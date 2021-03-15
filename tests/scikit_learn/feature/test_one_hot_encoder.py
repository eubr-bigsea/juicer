from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import OneHotEncoderOperation
from sklearn.preprocessing import OneHotEncoder
from tests.scikit_learn.util import get_X_train_data
from textwrap import dedent
import pytest
import pandas as pd
import numpy as np


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# OneHotEncoder
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_one_hot_encoder_success():
    df = util.iris(['sepalwidth', 'petalwidth'], size=10)
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
    result = util.execute(util.get_complete_code(instance), {'df': df})

    X_train = get_X_train_data(test_df, ['sepalwidth', 'petalwidth'])
    enc = OneHotEncoder()
    test_df['onehotenc_1'] = enc.fit_transform(X_train).toarray().tolist()
    assert result['out'].equals(test_df)
    assert str(enc) == str(result['enc'])
    assert dedent("""
    out = df
    enc = OneHotEncoder()
    X_train = get_X_train_data(df, ['sepalwidth', 'petalwidth'])
    out['onehotenc_1'] = enc.fit_transform(X_train).toarray().tolist()
    """) == instance.generate_code()


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
    result = util.execute(util.get_complete_code(instance), {'df': df})

    X_train = get_X_train_data(test_df,
                               ['sepalwidth', 'petalwidth', 'sepallength',
                                'petallength', 'class'])
    enc = OneHotEncoder()
    test_df['onehotenc_1'] = enc.fit_transform(X_train).toarray().tolist()
    assert result['out'].equals(test_df)
    assert str(enc) == str(result['enc'])
    assert dedent("""
    out = df
    enc = OneHotEncoder()
    X_train = get_X_train_data(df, ['sepalwidth', 'petalwidth', 'sepallength', 'petallength', 'class'])
    out['onehotenc_1'] = enc.fit_transform(X_train).toarray().tolist()
    """) == instance.generate_code()


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
    result = util.execute(util.get_complete_code(instance), {'df': df})
    X_train = get_X_train_data(test_df, ['sepalwidth', 'petalwidth'])
    enc = OneHotEncoder()
    test_df['onehotenc_1'] = enc.fit_transform(X_train).toarray().tolist()
    assert result['out'].equals(test_df)
    assert str(enc) == str(result['enc'])
    assert dedent("""
    out = df
    enc = OneHotEncoder()
    X_train = get_X_train_data(df, ['sepalwidth', 'petalwidth'])
    out['onehotenc_1'] = enc.fit_transform(X_train).toarray().tolist()
    """) == instance.generate_code()


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
    result = util.execute(util.get_complete_code(instance), {'df': df})
    X_train = get_X_train_data(test_df, ['sepalwidth', 'petalwidth'])
    enc = OneHotEncoder()
    test_df['onehotenc_1'] = enc.fit_transform(X_train).toarray().tolist()
    assert result['out'].equals(test_df)
    assert str(enc) == str(result['enc'])
    assert dedent("""
    out = df
    enc = OneHotEncoder()
    X_train = get_X_train_data(df, ['sepalwidth', 'petalwidth'])
    out['onehotenc_1'] = enc.fit_transform(X_train).toarray().tolist()
    """) == instance.generate_code()


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
    result = util.execute(util.get_complete_code(instance), {'df': df})
    X_train = get_X_train_data(test_df, ['sepalwidth', 'petalwidth'])
    enc = OneHotEncoder()
    test_df['test_result'] = enc.fit_transform(X_train).toarray().tolist()
    assert result['out'].equals(test_df)
    assert str(enc) == str(result['enc'])
    assert dedent("""
    out = df
    enc = OneHotEncoder()
    X_train = get_X_train_data(df, ['sepalwidth', 'petalwidth'])
    out['test_result'] = enc.fit_transform(X_train).toarray().tolist()
    """) == instance.generate_code()


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
    assert f"Parameters 'attributes' must be informed for task" \
           f" {OneHotEncoderOperation}" in str(val_err.value)
