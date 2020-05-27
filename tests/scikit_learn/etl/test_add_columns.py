from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import AddColumnsOperation
import pandas as pd
import pytest


# Add columns operation
#
def test_add_columns_success():
    slice_size = 10
    left_df = ['df1', util.iris(['sepallength', 'sepalwidth'], slice_size)]
    right_df = ['df2', util.iris(['petallength', 'petalwidth', 'class'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': left_df[0],
            'input data 2': right_df[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': left_df[1], 'df2': right_df[1]})

    assert result['out'].equals(util.iris(size=slice_size))


def test_add_columns_fail_different_row_number():
    """
    In this case, AddColumnsOperation() creates a result
    the size of the smallest slice_size.
    """
    slice_size_1 = 10
    slice_size_2 = 5
    left_df = ['df1', util.iris(['sepallength', 'sepalwidth'], slice_size_1)]
    right_df = ['df2', util.iris(['petallength', 'petalwidth', 'class'], slice_size_2)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': left_df[0],
            'input data 2': right_df[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': left_df[1], 'df2': right_df[1]})

    with pytest.raises(AssertionError) as different_row_number:
        assert len(result['out']) == slice_size_1
    print(different_row_number)


def test_add_columns_success_same_column_names_with_parameter():
    slice_size = 10
    left_df = ['df1', util.iris(['sepallength', 'sepalwidth'], slice_size)]
    right_df = ['df2', util.iris(['sepallength', 'sepalwidth'], slice_size)]

    arguments = {
        'parameters': {AddColumnsOperation.ALIASES_PARAM: '_value0,_value1'},
        'named_inputs': {
            'input data 1': left_df[0],
            'input data 2': right_df[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': left_df[1], 'df2': right_df[1]})

    x = pd.merge(left_df[1], right_df[1], left_index=True,
                 right_index=True, suffixes=('_value0', '_value1'))

    assert result['out'].equals(x)


def test_add_columns_success_same_column_names_no_parameter():
    slice_size = 10
    left_df = ['df1', util.iris(['sepallength', 'class'], slice_size)]
    right_df = ['df2', util.iris(['sepallength', 'class'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': left_df[0],
            'input data 2': right_df[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AddColumnsOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': left_df[1], 'df2': right_df[1]})

    x = pd.merge(left_df[1], right_df[1], left_index=True,
                 right_index=True, suffixes=('_ds0', '_ds1'))

    assert result['out'].equals(x)


def test_add_columns_success_no_output_implies_no_code():
    slice_size = 10
    left_df = ['df1', util.iris(['sepallength', 'sepalwidth'], slice_size)]
    right_df = ['df2', util.iris(['petallength', 'petalwidth', 'class'],
                                 slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': left_df[0],
            'input data 2': right_df[0]
        },
        'named_outputs': {}
    }
    instance = AddColumnsOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': left_df[1], 'df2': right_df[1]})

    assert not instance.has_code
