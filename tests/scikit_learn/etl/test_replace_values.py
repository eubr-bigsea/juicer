from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import ReplaceValuesOperation
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# ReplaceValues
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_replace_values_success():
    df = util.iris(['class'], size=10)

    test_df = df.copy()
    test_df.loc[:, 'class'] = 'replaced'

    arguments = {
        'parameters': {'value': 'Iris-setosa', 'replacement': 'replaced',
                       'attributes': ['class']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ReplaceValuesOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].equals(test_df)


def test_replace_values_multiple_attributes_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    df.loc[5, ['sepallength', 'sepalwidth']] = 10
    test_df.loc[5, ['sepallength', 'sepalwidth']] = 'test'

    arguments = {
        'parameters': {'value': '10', 'replacement': 'test',
                       'attributes': ['sepallength', 'sepalwidth']
                       },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ReplaceValuesOperation(**arguments)
    result = util.execute(instance.generate_code(), {'df': df})
    assert result['out'].equals(test_df)


def test_replace_values_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'value': 'Iris-setosa', 'replacement': 'replaced',
                       'attributes': ['class']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = ReplaceValuesOperation(**arguments)
    assert instance.generate_code() is None


def test_replace_values_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'value': 'Iris-setosa', 'replacement': 'replaced',
                       'attributes': ['class']},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ReplaceValuesOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_replace_values_missing_replacement_param_fail():
    arguments = {
        'parameters': {'value': 'Iris-setosa',
                       'attributes': ['class']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        ReplaceValuesOperation(**arguments)
    assert "Parameter value and replacement must be informed if is using replace" \
           " by value in task" in str(val_err.value)


def test_replace_values_missing_value_param_fail():
    arguments = {
        'parameters': {'replacement': 'replaced',
                       'attributes': ['class']},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        ReplaceValuesOperation(**arguments)
    assert "Parameter value and replacement must be informed if is using" \
           " replace by value in task" in str(val_err.value)


def test_replace_values_missing_attributes_param_fail():
    arguments = {
        'parameters': {'value': 'Iris-setosa', 'replacement': 'replaced'},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(KeyError) as key_err:
        ReplaceValuesOperation(**arguments)
    assert "'attributes'" in str(key_err.value)
