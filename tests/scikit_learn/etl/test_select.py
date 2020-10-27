from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SelectOperation
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# Select test
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_select_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    arguments = {
        'parameters': {
            'attributes': ['sepallength']
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].equals(util.iris(['sepallength'], size=10))


def test_select_with_two_valid_attributes_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)

    arguments = {
        'parameters': {
            'attributes': ['sepallength', 'sepalwidth']
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].equals(
        util.iris(['sepallength', 'sepalwidth'], size=10))


def test_select_two_equal_parameters_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)

    arguments = {
        'parameters': {
            'attributes': ['sepallength', 'sepallength']
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].equals(
        util.iris(['sepallength', 'sepallength'], size=10))


def test_select_no_output_inplies_no_code_success():
    arguments = {
        'parameters': {
            'attributes': ['class']
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = SelectOperation(**arguments)
    assert instance.generate_code() is None


def test_select_missing_input_inplies_no_code_success():
    arguments = {
        'parameters': {
            'attributes': ['class']
        },
        'named_inputs': {},
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_select_missing_parameters_fail():
    arguments = {
        'parameters': {
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        SelectOperation(**arguments)
    assert "'attributes' must be informed for task" in str(val_err.value)


def test_select_invalid_and_valid_attributes_param_fail():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)

    arguments = {
        'parameters': {
            'attributes': ['sepallength', 'class']
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "['class'] not in index" in str(key_err.value)


def test_select_fail_invalid_attributes_param_fail():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)

    arguments = {
        'parameters': {
            'attributes': ['class']
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "None of [Index(['class'], dtype='object')] are in the [columns]" in str(
        key_err.value)


def test_select_two_invalid_attributes_param_fail():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)

    arguments = {
        'parameters': {
            'attributes': ['class', 'class2']
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "None of [Index(['class', 'class2'], dtype='object')] are in the" \
           " [columns]" in str(key_err.value)


def test_select_fail_invalid_named_inputs():
    arguments = {
        'parameters': {
            'attributes': ['sepallength']
        },
        'named_inputs': {
            'input data': 'error',
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    with pytest.raises(TypeError) as typ_err:
        util.execute(instance.generate_code(),
                     {'error': []})
    assert 'list indices must be integers or slices, not list' in str(
        typ_err.value)


def test_select_two_named_inputs_fail():
    arguments = {
        'parameters': {
            'attributes': ['sepallength']
        },
        'named_inputs': {
            'input data': ['error', 'error1'],
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    with pytest.raises(TypeError) as typ_err:
        util.execute(instance.generate_code(),
                     {'error': []} and {'error1': []})
    assert 'list indices must be integers or slices, not list' in str(
        typ_err.value)
