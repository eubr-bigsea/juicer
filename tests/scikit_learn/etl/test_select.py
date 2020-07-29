from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SelectOperation
import pytest

# Select test


def test_select_success():
    slice_size = 10
    df =['df', util.iris(['sepallength', 'sepalwidth',
        'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {
            SelectOperation.ATTRIBUTES_PARAM: ['sepallength']
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    """
    print("\n==========generated code============\n")
    print(instance.generate_code())
    print("\n=====================================\n")
    """
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    #print(result)

    assert result['out'].equals(util.iris(['sepallength'],size=slice_size))

def test_select_fail_missing_parameters():
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
        instance = SelectOperation(**arguments)
    assert "'attributes' must be informed for task" in str(exc_info.value)


def test_select_success_with_2_valid_parameters():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {
            SelectOperation.ATTRIBUTES_PARAM: ['sepallength','sepalwidth']
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    """
    print("\n==========generated code============\n")
    print(instance.generate_code())
    print("\n=====================================\n")
    """
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    #print(result)

    assert result['out'].equals(util.iris(['sepallength','sepalwidth'], size=slice_size))


def test_select_fail_invalid_and_valid_parameters():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {
            SelectOperation.ATTRIBUTES_PARAM: ['sepallength','class']
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    with pytest.raises(KeyError) as exc_info:
        result = util.execute(instance.generate_code(),
                              {'df': df[1]})
    """
    print("\n==========generated code============\n")
    print(instance.generate_code())
    print("\n=====================================\n")
    """
    #print(exc_info.value)
    assert "['class'] not in index" in str(exc_info.value)

def test_select_fail_invalid_parameter():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {
            SelectOperation.ATTRIBUTES_PARAM: ['class']
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    with pytest.raises(KeyError) as exc_info:
        result = util.execute(instance.generate_code(),
                      {'df': df[1]})
    """
    print("\n==========generated code============\n")
    print(instance.generate_code())
    print("\n=====================================\n")
    """
    #print(exc_info.value)
    assert "None of [Index(['class'], dtype='object')] are in the [columns]" in str(exc_info.value)

def test_select_fail_2_invalid_parameters():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {
            SelectOperation.ATTRIBUTES_PARAM: ['class','class2']
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    with pytest.raises(KeyError) as exc_info:
        result = util.execute(instance.generate_code(),
                              {'df': df[1]})
    """
    print("\n==========generated code============\n")
    print(instance.generate_code())
    print("\n=====================================\n")
    """
    #print(exc_info.value)
    assert "None of [Index(['class', 'class2'], dtype='object')] are in the [columns]" in str(exc_info.value)

def test_select_success_2_equal_parameters():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {
            SelectOperation.ATTRIBUTES_PARAM: ['sepallength','sepallength']
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    """
    print("\n==========generated code============\n")
    print(instance.generate_code())
    print("\n=====================================\n")
    """
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    #print(result)

    assert result['out'].equals(util.iris(['sepallength','sepallength'], size=slice_size))


def test_select_fail_invalid_named_inputs():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]
    er = ['error',[]]
    arguments = {
        'parameters': {
            SelectOperation.ATTRIBUTES_PARAM: ['sepallength']
        },
        'named_inputs': {
            'input data': er[0],
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    """
    print("\n==========generated code============\n")
    print(instance.generate_code())
    print("\n=====================================\n")
    """
    with pytest.raises(TypeError) as exc_info:
        result = util.execute(instance.generate_code(),
                          {'error': er[1]})

    #print(exc_info.value)
    assert 'list indices must be integers or slices, not list' in str(exc_info.value)


def test_select_fail_2_named_inputs():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth',
                           'petalwidth', 'petallength'], slice_size)]
    er = ['error', []]
    er1 = ['error1', []]
    arguments = {
        'parameters': {
            SelectOperation.ATTRIBUTES_PARAM: ['sepallength']
        },
        'named_inputs': {
            'input data': [er[0],er1[0]],
        },
        'named_outputs': {
            'output projected data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    """
    print("\n==========generated code============\n")
    print(instance.generate_code())
    print("\n=====================================\n")
    """
    with pytest.raises(TypeError) as exc_info:
        result = util.execute(instance.generate_code(),
                              {'error': er[1]} and {'error1': er1[1]})

    #print(exc_info.value)
    assert 'list indices must be integers or slices, not list' in str(exc_info.value)


def test_select_success_missing_input_inplies_no_code():
    #deve dar sucessso porem sem gerar codigo, has_code() deve retornar False
    slice_size = 10
    df = ['df', util.iris(size=slice_size)]

    arguments = {
        'parameters': {
            SelectOperation.ATTRIBUTES_PARAM: ['class']
        },
        'named_inputs': {},
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SelectOperation(**arguments)
    assert not instance.has_code


def test_select_success_missing_output_inplies_no_code():
    #deve dar sucessso porem sem gerar codigo
    slice_size = 10
    df = ['df', util.iris(size=slice_size)]

    arguments = {
        'parameters': {
            SelectOperation.ATTRIBUTES_PARAM: ['class']
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
        }
    }
    instance = SelectOperation(**arguments)
    """
    print("\n==========generated code============\n")
    print(instance.generate_code())
    print("\n=====================================\n")
    """
    assert not instance.has_code
