from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import MaxAbsScalerOperation
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


def scaler(df, cols):
    return [[df.loc[idx, col] / df[col].max() for col in cols]
            for idx in df[cols].index]


# MaxAbsScaler
#
#
# # # # # # # # # # Success # # # # # # # # # #

def test_max_abs_scaler_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'attribute': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MaxAbsScalerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    data = {'scaled_1': scaler(df, ['sepalwidth', 'petalwidth'])}
    data = pd.DataFrame(data)

    assert test_df.max()['sepalwidth'] == 3.9
    assert test_df.max()['petalwidth'] == 0.4
    assert result['out'].loc[:, 'scaled_1'].equals(data.loc[:, 'scaled_1'])


def test_max_abs_scaler_alias_param_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    arguments = {
        'parameters': {'attribute': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0},
                       'alias': 'test_pass'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MaxAbsScalerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].iloc[:, 2].name == 'test_pass'


def test_max_abs_scaler_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attribute': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = MaxAbsScalerOperation(**arguments)
    assert instance.generate_code() is None


def test_max_abs_scaler_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attribute': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MaxAbsScalerOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_max_abs_scaler_missing_attributes_param_fail():
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
        MaxAbsScalerOperation(**arguments)
    assert "Parameters 'attribute' must be informed for task" in str(
        val_err.value)
