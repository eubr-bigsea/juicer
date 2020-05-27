from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import AddColumnsOperation
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

    It doesn't look like a bug, but the test name is
    test_add_columns_'fail'_different_row_number
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


def test_add_columns_success_same_column_names_but_with_suffixes():
    slice_size = 10
    left_df = ['df1', util.iris(['sepallength', 'sepalwidth'], slice_size)]
    right_df = ['df2', util.iris(['sepallength', 'sepalwidth'], slice_size)]

    arguments = {
        'parameters': {AddColumnsOperation.ALIASES_PARAM: '_ds0,_ds1'},
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

    assert not result['out'].equals(left_df[1])
    assert not result['out'].equals(right_df[1])


def test_add_columns_fail_same_column_names():
    """
    In this case, AddColumnsOperation() creates
    a result with duplicated columns names
    """
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
    with pytest.raises(AssertionError) as same_column_names:
        assert result['out'].equals(left_df[1])
        assert result['out'].equals(right_df[1])
    print(same_column_names)


def test_add_columns_success_using_prefix():
    slice_size = 10
    left_df = ['df1', util.iris(['sepallength', 'sepalwidth'], slice_size)]
    right_df = ['df2', util.iris(['sepallength', 'sepalwidth'], slice_size)]

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

    result['out'] = result['out'].add_prefix('col_')

    assert not result['out'].equals(left_df[1])
    assert not result['out'].equals(right_df[1])


def no_output():
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


def test_add_columns_success_no_output_implies_no_code():
    with pytest.raises(ValueError) as no_out:
        no_output()
    print(f'\n\n{no_out}')
