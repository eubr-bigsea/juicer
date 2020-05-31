from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import UnionOperation
import pandas as pd
import pytest


# Union
# 
def test_union_success():
    slice_size = 10
    df1 = ['df1', util.iris(['sepallength', 'sepalwidth',
                             'petalwidth', 'petallength'], slice_size)]
    df2 = ['df2', util.iris(['sepallength', 'sepalwidth',
                             'petalwidth', 'petallength'], slice_size)]
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = UnionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1], 'df2': df2[1]})

    x = pd.concat([df1[1], df2[1]], sort=False, axis=0, ignore_index=True)
    assert result['out'].equals(x)
    assert len(result['out']) == 20


def test_union_fail():
    slice_size = 10
    df1 = ['df1', util.iris(['sepallength', 'sepalwidth',
                             'petalwidth', 'petallength'], slice_size)]
    df2 = ['df2', util.iris(['sepallength', 'sepalwidth',
                             'petalwidth', 'petallength'], slice_size)]
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = UnionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1], 'df2': df2[1]})

    x = pd.concat([df1[1], df2[1]], sort=False, axis=0, ignore_index=True)
    with pytest.raises(AssertionError) as assertion_error:
        assert not result['out'].equals(x)
        assert not len(result['out']) == 20
    print(assertion_error)


def test_union_success_different_columns():
    slice_size = 10
    df1 = ['df1', util.iris(['sepallength', 'sepalwidth'], slice_size)]
    df2 = ['df2', util.iris(['petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = UnionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1], 'df2': df2[1]})

    x = pd.concat([df1[1], df2[1]], sort=False, axis=0, ignore_index=True)
    assert result['out'].equals(x)
    assert len(result['out']) == 20


def no_output():
    slice_size = 10
    df1 = ['df', util.iris(['sepallength', 'sepalwidth',
                            'petallength', 'petalwidth'], slice_size)]
    df2 = ['df', util.iris(['sepallength', 'sepalwidth',
                            'petallength', 'petalwidth'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data 1': df1[0],
            'inpút data 2': df2[0]
        },
        'named_outputs': {}
    }
    instance = UnionOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df1': df1[1], 'df2': df2[1]})


def test_union_success_no_output_implies_no_code():
    with pytest.raises(ValueError) as no_output_code:
        no_output()
    print(no_output_code)
