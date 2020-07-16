import pandas as pd
from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SplitOperation
import pytest


# Split
#
def test_split_data_integrity():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petallength', 'petalwidth'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    result = [result['split_1_task_1'], result['split_2_task_1']]
    result = pd.concat(result, axis=0, join='outer', ignore_index=False, keys=None,
                       levels=None, names=None, verify_integrity=False, copy=True)
    result = result.sort_index()

    assert len(result) == slice_size
    assert result.equals(df[1])


def test_split_randomness():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petallength', 'petalwidth'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    result = [result['split_1_task_1'], result['split_2_task_1']]
    result = pd.concat(result, axis=0, join='outer', ignore_index=False, keys=None,
                       levels=None, names=None, verify_integrity=False, copy=True)

    assert not result.equals(df[1])


def test_split_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petallength', 'petalwidth'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert len(result['split_1_task_1']) == 5
    assert len(result['split_2_task_1']) == 5


def test_split_success_one_slice():
    slice_size = 1
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petallength', 'petalwidth'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert len(result['split_2_task_1']) == slice_size


def test_split_fail_by_seed():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petallength', 'petalwidth'], slice_size)]

    arguments = {
        'parameters': {'seed': 4294967296  # Seeds higher or equal to 4294967296, and lower than 0 break the code
                       },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    with pytest.raises(ValueError) as invalid_seed:
        result = util.execute(instance.generate_code(),
                              {'df': df[1]})
    print(invalid_seed)


def test_weird_split_by_weights_and_slice():
    """
    This happens when the 'slice_size' is between 1 and 10
    and the 'weights' value is between -9 and 9

    The weights don't make a difference in these cases

    I don't know if it's a bug or not
    """
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petallength', 'petalwidth'], slice_size)]

    arguments = {
        'parameters': {'weights': 9},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert len(result['split_2_task_1']) == slice_size


def test_split_success_no_output_implies_no_code():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petallength', 'petalwidth'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {}
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert not instance.has_code
