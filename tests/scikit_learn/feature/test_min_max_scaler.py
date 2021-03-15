from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import MinMaxScalerOperation
from juicer.scikit_learn.util import get_X_train_data
from textwrap import dedent
from sklearn.preprocessing import MinMaxScaler
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# MinMaxScaler
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_min_max_scaler_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0},
                       'min': 0,
                       'max': 1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MinMaxScalerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    cols = ['sepalwidth', 'petalwidth']
    X_train = get_X_train_data(test_df, cols)
    model_1 = MinMaxScaler(feature_range=(0, 1))
    values = model_1.fit_transform(X_train)
    test_out = pd.concat(
        [test_df, pd.DataFrame(
            values, columns=['sepalwidth_norm',
                             'petalwidth_norm'])], ignore_index=False, axis=1)
    assert result['out'].equals(test_out)
    assert str(result['model_1']) == str(model_1)
    assert instance.generate_code() == dedent(
        """
    X_train = get_X_train_data(df, ['sepalwidth', 'petalwidth'])
    
    model_1 = MinMaxScaler(feature_range=(0,1))
    values = model_1.fit_transform(X_train)
    
    out = pd.concat([df, 
        pd.DataFrame(values, columns=['sepalwidth_norm', 'petalwidth_norm'])],
        ignore_index=False, axis=1)
        """)


def test_min_max_scaler_2_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0},
                       'min': 2,
                       'max': 6},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MinMaxScalerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    cols = ['sepalwidth', 'petalwidth']
    X_train = get_X_train_data(test_df, cols)
    model_1 = MinMaxScaler(feature_range=(2, 6))
    values = model_1.fit_transform(X_train)
    test_out = pd.concat(
        [test_df, pd.DataFrame(
            values, columns=['sepalwidth_norm',
                             'petalwidth_norm'])], ignore_index=False, axis=1)
    assert result['out'].equals(test_out)
    assert str(result['model_1']) == str(model_1)
    assert instance.generate_code() == dedent(
        """
    X_train = get_X_train_data(df, ['sepalwidth', 'petalwidth'])

    model_1 = MinMaxScaler(feature_range=(2,6))
    values = model_1.fit_transform(X_train)

    out = pd.concat([df, 
        pd.DataFrame(values, columns=['sepalwidth_norm', 'petalwidth_norm'])],
        ignore_index=False, axis=1)
        """)


def test_min_max_scaler_alias_param_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0},
                       'min': 0,
                       'max': 1,
                       'alias': 'success1, success2'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MinMaxScalerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    cols = ['sepalwidth', 'petalwidth']
    X_train = get_X_train_data(test_df, cols)
    model_1 = MinMaxScaler(feature_range=(0, 1))
    values = model_1.fit_transform(X_train)
    test_out = pd.concat(
        [test_df, pd.DataFrame(
            values, columns=['success1',
                             'success2'])], ignore_index=False, axis=1)


def test_min_max_scaler_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0},
                       'min': 0,
                       'max': 1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = MinMaxScalerOperation(**arguments)
    assert instance.generate_code() is None


def test_min_max_scaler_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0},
                       'min': 0,
                       'max': 1},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MinMaxScalerOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_min_max_scaler_missing_attribute_param_fail():
    arguments = {
        'parameters': {'multiplicity': {'input data': 0},
                       'min': 0,
                       'max': 1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        MinMaxScalerOperation(**arguments)
    assert f"Parameters 'attributes' must be informed for task" \
           f" {MinMaxScalerOperation}" in str(val_err.value)
