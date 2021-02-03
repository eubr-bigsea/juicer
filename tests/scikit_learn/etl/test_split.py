import pandas as pd
from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SplitOperation
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# Split
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_split_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petallength', 'petalwidth'], size=10)

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    assert len(result['split_1_task_1']) == 5
    assert len(result['split_2_task_1']) == 5


def test_split_uneven_size_dataframe_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petallength', 'petalwidth'], size=13)

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    assert len(result['split_1_task_1']) == 6
    assert len(result['split_2_task_1']) == 7


def test_split_seed_param_success():
    """
    Seeds higher than the integer limit and lower than zero will be set to 0
    """
    df = util.iris(['sepallength', 'sepalwidth',
                    'petallength', 'petalwidth'], size=10)
    test_out = df.copy()
    test_out.index = [8, 4, 0, 7, 2, 9, 5, 6, 1, 3]
    test_out.sort_index(axis=0, inplace=True)
    test_out.index = [2, 8, 4, 9, 1, 6, 7, 3, 0, 5]
    arguments = {
        'parameters': {'seed': -1},
        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    assert len(result['split_1_task_1']) == 5
    assert len(result['split_2_task_1']) == 5
    assert test_out.iloc[:5, :].equals(result['split_1_task_1'])
    assert test_out.iloc[5:10, :].equals(result['split_2_task_1'])


def test_split_randomness_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petallength', 'petalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = pd.concat(
        [result['split_1_task_1'], result['split_2_task_1']])
    assert not test_out.equals(test_df)


def test_split_data_integrity_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petallength', 'petalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = pd.concat(
        [result['split_1_task_1'], result['split_2_task_1']]).sort_index()
    assert test_out.equals(test_df)


def test_split_one_row_dataframe_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petallength', 'petalwidth'], size=1)
    test_df = df.copy()

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['split_2_task_1'].equals(test_df)
    assert len(result['split_2_task_1']) == 1


def test_split_weights_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petallength', 'petalwidth'], size=50)

    arguments = {
        'parameters': {'weights': 36},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    assert len(result['split_1_task_1']) == 18
    assert len(result['split_2_task_1']) == 32


def test_split_no_output_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = SplitOperation(**arguments)
    assert instance.generate_code() is None


def test_split_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {},
        'named_inputs': {
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_split_invalid_seed_param_fail():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petallength', 'petalwidth'], size=10)

    arguments = {
        'parameters': {'seed': 'invalid'},
        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    instance = SplitOperation(**arguments)
    with pytest.raises(NameError) as nam_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "name 'invalid' is not defined" in str(nam_err.value)


def test_split_invalid_weights_param_fail():
    arguments = {
        'parameters': {'weights': 'invalid'},
        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'split 1': 'split_1_task_1',
            'split 2': 'split_2_task_1'
        }
    }
    with pytest.raises(ValueError) as val_err:
        SplitOperation(**arguments)
    assert "could not convert string to float: 'invalid'" in str(
        val_err.value)
