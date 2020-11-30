from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import DropOperation
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# Drop
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_drop_success():
    df = util.iris(size=10)
    arguments = {
        'parameters': {
            'attributes': ['class']
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DropOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].equals(
        util.iris(size=10).drop(['class'], axis=1))


def test_drop_no_output_implies_no_code_success():
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
    instance = DropOperation(**arguments)
    assert instance.generate_code() is None


def test_drop_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {
            'attributes': ['class']
        },
        'named_inputs': {},
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DropOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_drop_missing_parameters_fail():
    df = util.iris(size=10)

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
        DropOperation(**arguments)
    assert "'attributes' must be informed for task" in str(val_err.value)


def test_drop_invalid_attribute_param_fail():
    df = util.iris(size=10)

    arguments = {
        'parameters': {
            'attributes': ['invalid']
        },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DropOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "['invalid'] not found in axis" in str(key_err.value)
