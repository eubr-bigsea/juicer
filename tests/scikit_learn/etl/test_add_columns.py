from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import AddColumnsOperation
import pandas as pd
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# Add columns operation
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_add_columns_success():
    left_df = util.iris(['sepallength', 'sepalwidth'], size=10)
    right_df = util.iris(['petallength', 'petalwidth', 'class'], size=10)
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': left_df, 'df2': right_df})
    assert result['out'].equals(util.iris([
        'sepallength', 'sepalwidth',
        'petallength', 'petalwidth', 'class'], size=10))


def test_add_columns_different_size_dataframes_success():
    left_df = util.iris(['sepallength', 'sepalwidth'], size=10)
    right_df = util.iris(['petallength', 'petalwidth', 'class'], size=5)
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': left_df, 'df2': right_df})
    assert result['out'].equals(util.iris([
        'sepallength', 'sepalwidth',
        'petallength', 'petalwidth', 'class'], size=5))


def test_add_columns_aliases_param_success():
    left_df = util.iris(['sepallength', 'sepalwidth'], size=10)
    right_df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = util.iris(
        ['sepallength', 'sepalwidth', 'sepallength', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'aliases': '_value0,_value1'},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': left_df, 'df2': right_df})
    test_df.columns = ['sepallength_value0', 'sepalwidth_value0',
                       'sepallength_value1', 'sepalwidth_value1']
    assert result['out'].equals(test_df)


def test_add_columns_repeated_column_names_success():
    left_df = util.iris(['sepallength', 'class'], size=10)
    right_df = util.iris(['sepallength', 'class'], size=10)
    test_df = util.iris(
        ['sepallength', 'class', 'sepallength', 'class'], size=10)
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': left_df, 'df2': right_df})
    test_df.columns = ['sepallength_ds0', 'class_ds0',
                       'sepallength_ds1', 'class_ds1']
    assert result['out'].equals(test_df)


def test_add_columns_no_output_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
        }
    }
    instance = AddColumnsOperation(**arguments)
    assert instance.generate_code() is None


def test_add_columns_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_add_columns_invalid_aliases_param_fail():
    left_df = util.iris(['sepallength', 'sepalwidth'], size=10)
    right_df = util.iris(['sepallength', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'aliases': 'invalid'},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    with pytest.raises(IndexError) as idx_err:
        util.execute(instance.generate_code(),
                     {'df1': left_df, 'df2': right_df})
    assert 'list index out of range' in str(idx_err.value)
