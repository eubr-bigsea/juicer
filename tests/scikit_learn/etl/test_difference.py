from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import DifferenceOperation
import pandas as pd
import numpy as np
import pytest


# Difference
#
def test_difference_one_col_success():
    df1 = ['df1', util.iris(['sepallength'], 15)]

    df2 = ['df2', util.iris(['sepallength'], 10)]

    df1[1].loc[7:9, 'sepallength'] = 1

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DifferenceOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})

    assert result['out'].equals(df1[1].loc[7:14, 'sepallength'].to_frame())
    assert len(result['out']) == 8


def test_difference_multiple_col_success():
    df1 = ['df1', util.iris(['sepallength', 'petalwidth'], 15)]

    df2 = ['df2', util.iris(['sepallength', 'petalwidth'], 10)]

    df1[1].loc[7:9, 'sepallength'] = 1

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DifferenceOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})
    assert result['out'].equals(df1[1].loc[7:14, ['sepallength', 'petalwidth']])
    assert len(result['out']) == 8


def test_difference_col_reorder_success():
    df1 = ['df1', util.iris(['sepallength', 'petalwidth', 'class'], 15)]

    df2 = ['df2', util.iris(['class', 'petalwidth', 'sepallength'], 10)]

    df1[1].loc[7:9, 'class'] = 'replaced'

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DifferenceOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})
    assert result['out'].equals(
        df1[1].loc[7:14, ['sepallength', 'petalwidth', 'class']])
    assert len(result['out']) == 8


def test_difference_col_intersection_success():
    df1 = ['df1', util.iris(['sepallength'], 15)]

    df2 = ['df2', util.iris(['petalwidth', 'sepallength'], 10)]

    df1[1].loc[7:9, 'sepallength'] = 1

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DifferenceOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})

    assert result['out'].equals(df1[1].loc[7:14, ['sepallength']])
    assert len(result['out']) == 8


def test_difference_col_intersection_2_success():
    df1 = ['df1', util.iris(['sepallength', 'petalwidth'], 15)]

    df2 = ['df2', util.iris(['petalwidth'], 10)]

    df1[1].loc[7:9, 'petalwidth'] = 1

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DifferenceOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})

    assert result['out'].equals(df1[1].loc[7:14, ['petalwidth']])
    assert len(result['out']) == 8


def test_difference_input2_is_bigger_success():
    # It only returns the dataframe column
    # Maybe it should return nothing?
    df1 = ['df1', util.iris(['petalwidth'], 10)]

    df2 = ['df2', util.iris(['petalwidth'], 15)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DifferenceOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})

    test = df1[1].drop(range(10))
    assert result['out'].equals(test)
    assert len(result['out']) == 0


def test_difference_different_cols_success():
    # Returns nothing
    df1 = ['df1', util.iris(['petalwidth'], 20)]
    df2 = ['df2', util.iris(['class'], 10)]

    arguments = {
        'parameters': {'attributes': 2},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DifferenceOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})

    test = df1[1].drop(columns='petalwidth', index=range(20))
    assert result['out'].equals(test)
    assert len(result['out']) == 0


def test_difference_big_variation_success():
    df1 = ['df1', util.iris(['petalwidth'], 40)]
    df2 = ['df2', util.iris(['petalwidth'], 10)]

    df1[1].loc[4, 'petalwidth'] = np.int64(50)
    df1[1].loc[5, 'petalwidth'] = pd.Timestamp(1596509236)
    df1[1].loc[6, 'petalwidth'] = np.float(1.56)
    df1[1].loc[7, 'petalwidth'] = np.array('test')
    df1[1].loc[8, 'petalwidth'] = np.bool(False)
    df1[1].loc[9, 'petalwidth'] = np.NaN

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DifferenceOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})
    # needs a better assertion
    assert len(result['out']) == 36


def test_difference_missing_inputs_success():
    df1 = ['df1', util.iris(['sepallength'], 15)]

    df2 = ['df2', util.iris(['sepallength'], 10)]

    df1[1].loc[7:9, 'sepallength'] = 1

    arguments = {
        'parameters': {},
        'named_inputs': {},
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(KeyError) as key_err:
        instance = DifferenceOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df1': df1[1],
                               'df2': df2[1]})
    assert 'input data 1' in str(key_err)


def test_difference_no_output_implies_no_code():
    df1 = ['df1', util.iris(['sepallength', 'petallength'], 15)]

    df2 = ['df2', util.iris(['sepallength', 'petallength'], 10)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {}
    }
    instance = DifferenceOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1],
                           'df2': df2[1]})
    assert not instance.has_code
