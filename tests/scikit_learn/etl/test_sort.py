from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SortOperation
import pytest
import pandas

# Sort
# 
import pytest
import pandas

# Sort
#
def test_sort_success():
    slice_size = 150
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {
            SortOperation.ATTRIBUTES_PARAM:[{'attribute': 'sepalwidth',
                                             'f':'sur'},
                                            ]},
        'named_inputs': {
            'input data': df[0],
            },
        'named_outputs': {
            'output data': 'out'
        }
    }
    pandas.set_option('display.max_rows', None)
    instance = SortOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          dict([df]))
    assert result != df
    assert slice_size == 150



def test_missing_parameters():
    slice_size = 150
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                            'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters':{},
        'named_inputs': {
            'input data': df[0],
            },
        'named_outputs': {
            'output data': 'out'
            }
        }
    pandas.set_option('display.max_rows', None)
    with pytest.raises(ValueError) as exc_info:
        instance = SortOperation(**arguments)
    assert "'attributes' must be informed for task" in str(exc_info.value)

def test_missing_named_inputs():
    slice_size = 150
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {
            SortOperation.ATTRIBUTES_PARAM: [{'attribute': 'sepallength',
                                              'f': 'sur'},
                                             ]},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    pandas.set_option('display.max_rows', None)
    instance = SortOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          dict([df]))
    assert SortOperation.has_code

def test_ascending_false():
    slice_size = 150
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {
            SortOperation.ATTRIBUTES_PARAM:[{'attribute': 'sepalwidth',
                                             'f':'false'},
                                            ]},
        'named_inputs': {
            'input data': df[0],
            },
        'named_outputs': {
            'output data': 'out'
        }
    }
    pandas.set_option('display.max_rows', None)
    instance = SortOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          dict([df]))

    assert instance.ascending == [False]


def test_ascending_true():
    slice_size = 150
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {
            SortOperation.ATTRIBUTES_PARAM: [{'attribute': 'sepalwidth',
                                              'f': 'asc'},
                                             ]},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    pandas.set_option('display.max_rows', None)
    instance = SortOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          dict([df]))

    assert instance.ascending == [True]
