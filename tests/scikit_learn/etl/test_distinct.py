from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import DistinctOperation
import pandas as pd
import numpy as np
import pytest


# Distinct
# 
def test_distinct_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength'], slice_size)]

    df[1].loc[0:3, 'sepallength'] = 'test'
    df[1].loc[6:9, 'sepallength'] = 'distinct'

    arguments = {
        'parameters': {'attributes': ['sepallength']},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DistinctOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    assert result['out'].equals(df[1].drop(index=[1, 2, 3, 7, 8, 9]))


def test_distinct_success_missing_parameters():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DistinctOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    assert result['out'].equals(df[1])


def test_distinct_success_wrong_attribute():
    slice_size = 10
    df = ['df', util.iris(['petallength'], slice_size)]

    df[1].loc[0:3, 'petallength'] = 10
    df[1].loc[6:9, 'petallength'] = 10

    arguments = {
        'parameters': {'attributes': ['sepallength']},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(KeyError) as key_err:
        instance = DistinctOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df': df[1]})
    assert "Index(['sepallength']" in str(key_err)


def test_distinct_success_missing_named_inputs():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    df[1].loc[0, ['sepallength', 'sepalwidth']] = 10
    df[1].loc[9, ['sepallength', 'sepalwidth']] = 10

    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth']},
        'named_inputs': {},
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(KeyError) as key_err:
        instance = DistinctOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df': df[1]})
    assert 'input data' in str(key_err)


def test_distinct_success_no_output_implies_no_code():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]
    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth']},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {}
    }
    instance = DistinctOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    assert not instance.has_code
