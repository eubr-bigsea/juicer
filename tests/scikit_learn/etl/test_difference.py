from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import DifferenceOperation
import pandas as pd
import numpy as np
import pytest

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# Difference
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_difference_one_col_success():
    df1 = util.iris(['sepallength'], size=15)
    df2 = util.iris(['sepallength'], size=10)
    df1.loc[5, 'sepallength'] = 9.9
    df1.loc[7:9, 'sepallength'] = 1.9

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
    instance = DifferenceOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1,
                           'df2': df2})

    assert len(result['out']) == 8


def test_difference_multiple_col_success():
    df1 = util.iris(['sepallength', 'petalwidth'], size=15)
    df2 = util.iris(['sepallength', 'petalwidth'], size=10)
    df1.loc[7:9, 'sepallength'] = 1

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
    instance = DifferenceOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1,
                           'df2': df2})

    output = df1.loc[7:, ['sepallength', 'petalwidth']].reset_index(drop=True)
    assert result['out'].equals(output)
    assert len(result['out']) == 8


def test_difference_col_reorder_success():
    df1 = util.iris(['sepallength', 'petalwidth', 'class'], size=15)
    df2 = util.iris(['class', 'petalwidth', 'sepallength'], size=10)
    df1.loc[7:9, 'class'] = 'replaced'
    test_df = df1.copy()

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
    instance = DifferenceOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1,
                           'df2': df2})
    output = test_df.loc[7:14, ['sepallength', 'petalwidth', 'class']]\
        .reset_index(drop=True)
    assert result['out'].equals(output)
    assert len(result['out']) == 8


def test_difference_input2_is_bigger_success():
    """It only returns the dataframe column, maybe it should return nothing?"""
    df1 = util.iris(['petalwidth'], size=10)
    df2 = util.iris(['petalwidth'], size=15)
    test_df = df1.copy().drop(range(10))

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
    instance = DifferenceOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1,
                           'df2': df2})

    assert result['out'].equals(test_df)
    assert len(result['out']) == 0


def test_difference_different_cols_fail():
    """Returns nothing"""
    df1 = util.iris(['petalwidth'], size=20)
    df2 = util.iris(['class'], size=10)

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
    instance = DifferenceOperation(**arguments)
    with pytest.raises(ValueError) as val_err:
        util.execute(instance.generate_code(),
                     {'df1': df1,
                      'df2': df2})

    assert "Both data need to have the same columns and data types" \
           in str(val_err.value)


def test_difference_different_datatype_fail():
    df1 = util.iris(['petalwidth'], size=40)
    df2 = util.iris(['petalwidth'], size=10)

    df1.loc[5, 'petalwidth'] = pd.Timestamp(1596509236)

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
    instance = DifferenceOperation(**arguments)

    with pytest.raises(ValueError) as val_err:
        util.execute(instance.generate_code(),
                     {'df1': df1,
                      'df2': df2})
    assert "Both data need to have the same columns and data types" \
           in str(val_err.value)


def test_difference_no_output_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
        }
    }
    instance = DifferenceOperation(**arguments)
    assert instance.generate_code() is None


def test_difference_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DifferenceOperation(**arguments)
    assert instance.generate_code() is None
