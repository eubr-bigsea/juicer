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

    x = [result['split_1_task_1'], result['split_2_task_1']]
    y = pd.concat(x, axis=0, join='outer', ignore_index=False, keys=None,
                  levels=None, names=None, verify_integrity=False, copy=True)
    z = y.sort_index()

    assert len(y) == slice_size
    assert z.equals(util.iris(['sepallength', 'sepalwidth',
                               'petallength', 'petalwidth'], size=slice_size))


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
    assert len(result['split_1_task_1']) and len(result['split_2_task_1']) != slice_size


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

    x = [result['split_1_task_1'], result['split_2_task_1']]
    y = pd.concat(x, axis=0, join='outer', ignore_index=False, keys=None,
                  levels=None, names=None, verify_integrity=False, copy=True)

    assert not y.equals(util.iris(['sepallength', 'sepalwidth',
                                   'petallength', 'petalwidth'], size=slice_size))


def test_split_sucess_one_slice():
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
        }
    }
    instance = SplitOperation(**arguments)
    with pytest.raises(ValueError) as InvalidSeed:
        result = util.execute(instance.generate_code(),
                              {'df': df[1]})
    print(InvalidSeed)


def test_weird_split_by_weights_and_slice():
    """
    This happens when the 'slice_size' is between 1 and 10
    and the 'weights' value is between -9 and 9
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
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    with pytest.raises(AssertionError) as Weird:
        assert len(result['split_1_task_1']) and len(result['split_2_task_1']) != slice_size
    print(Weird)


def test_split_success_no_output_implies_no_code():
    """
    Change SplitOperation: it shouldn't generate
    code if there is not output!
    Does this happen to every operation???
    """
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petallength', 'petalwidth'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    with pytest.raises(AssertionError) as HasCode:
        assert not instance.has_code
    print(HasCode)
