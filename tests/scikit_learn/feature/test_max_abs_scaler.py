from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import MaxAbsScalerOperation
from juicer.scikit_learn.util import get_X_train_data
from textwrap import dedent
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

def test_max_abs_scaler_infer_alias_success():
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
    instance = MaxAbsScalerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    X_train = get_X_train_data(test_df, ['sepalwidth', 'petalwidth'])
    model_1 = MaxAbsScaler()
    values = model_1.fit_transform(X_train)
    res_maxabs = pd.DataFrame(values, columns=['sepalwidth_norm',
                                               'petalwidth_norm'])

    assert result['out'][['sepalwidth_norm', 'petalwidth_norm']]\
        .equals(res_maxabs)
    assert str(result['model_1']) == str(model_1)
    assert dedent("""
    X_train = get_X_train_data(df, ['sepalwidth', 'petalwidth'])

    model_1 = MaxAbsScaler()
    values = model_1.fit_transform(X_train)

    out = pd.concat([df, 
        pd.DataFrame(values, columns=['sepalwidth_norm', 'petalwidth_norm'])],
        ignore_index=False, axis=1)
    """) == instance.generate_code()


def test_max_abs_scaler_one_alias_param_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    arguments = {
        'parameters': {'attributes': ['sepalwidth'],
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


def test_max_abs_scaler_multiple_alias_param_success():
    df = util.iris(['sepalwidth',
                    'petalwidth'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0},
                       'alias': 'test_pass1, test_pass2'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MaxAbsScalerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    X_train = get_X_train_data(test_df, ['sepalwidth', 'petalwidth'])
    model_1 = MaxAbsScaler()
    values = model_1.fit_transform(X_train)
    res_maxabs = pd.DataFrame(values, columns=['test_pass1',
                                               'test_pass2'])

    assert result['out'][['test_pass1', 'test_pass2']].equals(res_maxabs)


def test_max_abs_scaler_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
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
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
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
    assert f"Parameters 'attributes' must be informed for task" \
           f" {MaxAbsScalerOperation}" in str(val_err.value)
