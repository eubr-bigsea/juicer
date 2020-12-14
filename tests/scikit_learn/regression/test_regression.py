from tests.scikit_learn import util
from juicer.scikit_learn.regression_operation import RegressionOperation
import pandas as pd
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# Regression
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_regression_read_common_params_success():
    arguments = {
        'parameters': {'features': ['test'],
                       'label': ['test'],
                       'prediction': ['test']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RegressionOperation(**arguments)
    instance.read_common_params(arguments['parameters'])


def test_regression_get_outputs_names_success():
    arguments = {
        'parameters': {'features': ['test'],
                       'label': ['test'],
                       'prediction': ['test']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RegressionOperation(**arguments)
    assert instance.get_output_names() == "regression_algorithm_1"


def test_regression_get_data_out_names_success():
    arguments = {
        'parameters': {'features': ['test'],
                       'label': ['test'],
                       'prediction': ['test']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RegressionOperation(**arguments)
    assert instance.get_data_out_names() == ''


# # # # # # # # # # Fail # # # # # # # # # #
def test_regression_missing_features_param_fail():
    arguments = {
        'parameters': {'label': ['test'],
                       'prediction': ['test']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RegressionOperation(**arguments)
    with pytest.raises(ValueError) as val_err:
        instance.read_common_params(arguments['parameters'])
    assert "Parameters 'features' and 'label' must be informed for task" in \
           str(val_err.value)


def test_regression_missing_label_param_fail():
    arguments = {
        'parameters': {'features': ['test'],
                       'prediction': ['test']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RegressionOperation(**arguments)
    with pytest.raises(ValueError) as val_err:
        instance.read_common_params(arguments['parameters'])
    assert "Parameters 'features' and 'label' must be informed for task" in \
           str(val_err.value)
