from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import MinMaxScalerOperation
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


def scaler(df, cols, mi, ma, alias='scaled_1'):
    if mi >= ma:
        raise ValueError(f"min value ({mi}) needs to be lower than"
                         f" max value ({ma})")
    data = [[(df.loc[idx, col] - df[col].min()) / (df[col].max() - df[col].min())
             * (ma - mi) + mi for col in cols] for idx in df[cols].index]
    return pd.DataFrame({alias: data})


# MinMaxScaler
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_min_max_scaler_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attribute': ['sepalwidth', 'petalwidth'],
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
    test_out = scaler(test_df, ['sepalwidth', 'petalwidth'], 0, 1)
    assert result['out'].loc[:, 'scaled_1'].equals(test_out.loc[:, 'scaled_1'])


def test_min_max_scaler_2_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attribute': ['sepalwidth', 'petalwidth'],
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
    test_out = scaler(test_df, ['sepalwidth', 'petalwidth'], 2, 6)
    assert result['out'].loc[:, 'scaled_1'].equals(test_out.loc[:, 'scaled_1'])


def test_min_max_scaler_alias_param_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    arguments = {
        'parameters': {'attribute': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0},
                       'min': 0,
                       'max': 1,
                       'alias': 'success'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MinMaxScalerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].iloc[:, 2].name == "success"


def test_min_max_scaler_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attribute': ['sepalwidth', 'petalwidth'],
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
        'parameters': {'attribute': ['sepalwidth', 'petalwidth'],
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
    assert "Parameters 'attribute' must be informed for task" in str(
        val_err.value)


def test_min_max_scaler_invalid_attribute_param_fail():
    df = util.iris(['sepalwidth',
                    'petalwidth',
                    'class'], size=10)
    arguments = {
        'parameters': {'attribute': ['sepalwidth', 'petalwidth', 'class'],
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
    with pytest.raises(ValueError) as val_err:
        util.execute(util.get_complete_code(instance), {'df': df})
    assert "could not convert string to float: 'Iris-setosa'" in str(
        val_err.value)


def test_min_max_scaler_invalid_min_max_param_fail():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    arguments = {
        'parameters': {'attribute': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0},
                       'min': 1,
                       'max': 1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MinMaxScalerOperation(**arguments)
    with pytest.raises(ValueError) as val_err:
        util.execute(util.get_complete_code(instance), {'df': df})
    assert "Minimum of desired feature range must be smaller than maximum." \
           " Got (1, 1)." in str(val_err.value)
