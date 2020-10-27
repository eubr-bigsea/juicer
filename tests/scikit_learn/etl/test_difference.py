from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import DifferenceOperation
import pandas as pd
import numpy as np


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
    df1.loc[7:9, 'sepallength'] = 1
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
    assert result['out'].equals(test_df.loc[7:14, 'sepallength'].to_frame())
    assert len(result['out']) == 8


def test_difference_multiple_col_success():
    df1 = util.iris(['sepallength', 'petalwidth'], size=15)
    df2 = util.iris(['sepallength', 'petalwidth'], size=10)
    df1.loc[7:9, 'sepallength'] = 1
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
    assert result['out'].equals(test_df.loc[7:14, ['sepallength', 'petalwidth']])
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
    assert result['out'].equals(
        test_df.loc[7:14, ['sepallength', 'petalwidth', 'class']])
    assert len(result['out']) == 8


def test_difference_col_intersection_success():
    df1 = util.iris(['sepallength'], size=15)
    df2 = util.iris(['petalwidth', 'sepallength'], size=10)
    df1.loc[7:9, 'sepallength'] = 1
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

    assert result['out'].equals(test_df.loc[7:14, ['sepallength']])
    assert len(result['out']) == 8


def test_difference_col_intersection_2_success():
    df1 = util.iris(['sepallength', 'petalwidth'], size=15)
    df2 = util.iris(['petalwidth'], size=10)
    df1.loc[7:9, 'petalwidth'] = 1
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

    assert result['out'].equals(test_df.loc[7:14, ['petalwidth']])
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


def test_difference_different_cols_success():
    """Returns nothing"""
    df1 = util.iris(['petalwidth'], size=20)
    df2 = util.iris(['class'], size=10)
    test_df = df1.copy().drop(columns='petalwidth', index=range(20))

    arguments = {
        'parameters': {'attributes': 2},
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


def test_difference_big_variation_success():
    df1 = util.iris(['petalwidth'], size=40)
    df2 = util.iris(['petalwidth'], size=10)
    test_df = df1.copy()

    df1.loc[4, 'petalwidth'] = np.int64(50)
    df1.loc[5, 'petalwidth'] = pd.Timestamp(1596509236)
    df1.loc[6, 'petalwidth'] = np.float(1.56)
    df1.loc[7, 'petalwidth'] = np.array('test')
    df1.loc[8, 'petalwidth'] = np.bool(False)
    df1.loc[10, 'petalwidth'] = np.NaN

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

    diff_oper = df1.eq(df2)
    for i in range(40):
        if diff_oper.iloc[i, 0:].all():
            test_df.drop(i, inplace=True)
    assert result['out'].eq(test_df).equals(test_df.eq(result['out']))
    assert len(result['out']) == 35


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
