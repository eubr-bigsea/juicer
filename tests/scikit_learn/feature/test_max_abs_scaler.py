from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import MaxAbsScalerOperation
import pytest
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

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
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = MaxAbsScaler()

    assert test_df.max()['sepalwidth'] == 3.9
    assert test_df.max()['petalwidth'] == 0.4
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)
    assert """
model_1 = MaxAbsScaler()
X_train = get_X_train_data(df, ['sepalwidth', 'petalwidth'])
model_1.fit(X_train)

out = df
out['scaled_1'] = model_1.transform(X_train).tolist()
""" == \
           instance.generate_code()


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
    result = util.execute(util.get_complete_code(instance), {'df': df})
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
