from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import DistinctOperation
import pandas as pd
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# Distinct
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_distinct_success():
    df = util.iris(['sepallength'], size=10)
    df.loc[0:3, 'sepallength'] = 'test'
    df.loc[6:9, 'sepallength'] = 'distinct'
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['sepallength']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DistinctOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].equals(test_df.drop(index=[1, 2, 3, 7, 8, 9]))


def test_distinct_missing_attributes_param_success():
    df = util.iris(['sepallength'], size=10)
    df.loc[0:3, 'sepallength'] = 'test'
    df.loc[6:9, 'sepallength'] = 'distinct'
    test_df = df.copy()

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DistinctOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].equals(test_df.drop(index=[1, 2, 3, 7, 8, 9]))


def test_distinct_no_output_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = DistinctOperation(**arguments)
    assert instance.generate_code() is None


def test_distinct_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DistinctOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_distinct_invalid_attribute_param_fail():
    df = util.iris(['petallength'], 10)
    df.loc[0:3, 'petallength'] = 10
    df.loc[6:9, 'petallength'] = 10

    arguments = {
        'parameters': {'attributes': 'invalid'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DistinctOperation(**arguments)
    with pytest.raises(NameError) as nam_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "name 'invalid' is not defined" in str(nam_err.value)
