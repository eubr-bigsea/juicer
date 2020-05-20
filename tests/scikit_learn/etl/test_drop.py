from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import DropOperation
import pytest

# Drop
# 
def test_drop_success():
    slice_size = 10
    df = ['df', util.iris(size=slice_size)]

    arguments = {
        'parameters': {
            DropOperation.ATTRIBUTES_PARAM: ['class']
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DropOperation(**arguments)
    result = util.execute(instance.generate_code(), 
                          dict([df]))
    assert result['out'].equals(
            util.iris(size=slice_size).drop(['class'], axis=1))
def test_drop_fail_missing_parameter():
    slice_size = 10
    df = ['df', util.iris(size=slice_size)]

    arguments = {
        'parameters': {
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as exc_info:
        instance = DropOperation(**arguments)
    assert "'attributes' must be informed for task" in str(exc_info.value)

def test_drop_fail_invalid_attribute():
    slice_size = 10
    df = ['df', util.iris(size=slice_size)]

    arguments = {
        'parameters': {
            DropOperation.ATTRIBUTES_PARAM: ['invalid']
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as exc_info:
        instance = DropOperation(**arguments)
    assert "'attributes' must be informed for task" in str(exc_info.value)
    
def test_drop_success_missing_input_implies_no_code():
    slice_size = 10
    df = ['df', util.iris(size=slice_size)]

    arguments = {
        'parameters': {
        },
        'named_inputs': {},
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DropOperation(**arguments)
    assert not instance.has_code
    
def test_drop_success_no_output_implies_no_code():
    slice_size = 10
    df = ['df', util.iris(size=slice_size)]

    arguments = {
        'parameters': {
            DropOperation.ATTRIBUTES_PARAM: ['class']
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
        }
    }
    instance = DropOperation(**arguments)
    assert not instance.has_code
