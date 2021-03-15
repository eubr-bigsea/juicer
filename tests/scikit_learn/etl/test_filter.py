from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import FilterOperation
import pandas as pd
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# Filter
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_filter_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    df.iloc[0, 0] = 1
    test_df = df.copy()
    arguments = {
        'parameters': {'filter': [{'attribute': 'sepallength',
                                   'f': '<',
                                   'value': 'sepalwidth'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FilterOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert not result['out'].equals(test_df)
    assert result['out'].equals(test_df.iloc[0, :].to_frame().transpose())
    assert instance.generate_code() == """
out = df
out = out.query('sepallength < sepalwidth')"""


def test_filter_expression_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'expression': [{'tree': {'type': 'Literal',
                                                'value': 'sepallength'}}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FilterOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert not result['out'].equals(test_df)
    assert instance.generate_code() == """
out = df
out = out[out.apply(lambda row: 'sepallength', axis=1)]"""


def test_filter_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'filter': [{'attribute': 'sepallength',
                                   'f': '<',
                                   'value': 'sepalwidth'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = FilterOperation(**arguments)
    assert instance.generate_code() is None


def test_filter_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'filter': [{'attribute': 'sepallength',
                                   'f': '<',
                                   'value': 'sepalwidth'}]},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FilterOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_filter_missing_filter_or_expression_param_fail():
    arguments = {
        'parameters': {

        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        FilterOperation(**arguments)
    assert "Parameter 'filter' must be informed for task" in str(val_err.value)
