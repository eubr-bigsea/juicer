from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SortOperation
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# Sort
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_sort_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=150)
    test_df = df.copy()
    arguments = {
        'parameters': {
            'attributes': [{'attribute': 'sepalwidth',
                            'f': 'sur'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SortOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert not result['out'].equals(test_df)
    assert """out = df.sort_values(by=['sepalwidth'], ascending=[False])""" == \
           instance.generate_code()


def test_sort_ascending_param_true_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {
            'attributes': [{'attribute': 'sepalwidth',
                            'f': 'asc'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SortOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert not result['out'].equals(test_df)
    assert instance.ascending == [True]


def test_sort_ascending_param_false_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {
            'attributes': [{'attribute': 'sepalwidth',
                            'f': 'false'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SortOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert instance.ascending == [False]
    assert not result['out'].equals(test_df)


def test_sort_no_output_implies_no_code_success():
    arguments = {
        'parameters': {
            'attributes': [{'attribute': 'sepallength',
                            'f': 'sur'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = SortOperation(**arguments)
    assert instance.generate_code() is None


def test_sort_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {
            'attributes': [{'attribute': 'sepallength',
                            'f': 'sur'}]},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SortOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_sort_missing_attributes_param_fail():
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        SortOperation(**arguments)
    assert "'attributes' must be informed for task" in str(val_err.value)
