from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import ReplaceValuesOperation
import pytest


# ReplaceValues
#
def test_replace_values_success():
    slice_size = 10
    df = ['df', util.iris(['class'], slice_size)]

    df_tst = util.iris(['class'], slice_size)
    df_tst.loc[:, 'class'] = 'replaced'

    arguments = {
        'parameters': {'value': 'Iris-setosa', 'replacement': 'replaced',
                       'attributes': ['class']},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ReplaceValuesOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert result['out'].equals(df_tst)


def test_replace_values_multiple_attributes_success():
    slice_size = 10

    df = ['df', util.iris(['sepallength', 'sepalwidth'], slice_size)]
    df_tst = util.iris(['sepallength', 'sepalwidth'], slice_size)

    df[1].loc[5, ['sepallength', 'sepalwidth']] = 10
    df_tst.loc[5, ['sepallength', 'sepalwidth']] = 'test'

    arguments = {
        'parameters': {'value': '10', 'replacement': 'test',
                       'attributes': ['sepallength', 'sepalwidth']
                       },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ReplaceValuesOperation(**arguments)
    result = util.execute(instance.generate_code(), {'df': df[1]})

    assert result['out'].equals(df_tst)


def test_replace_missing_parameters_success():
    slice_size = 10
    df = ['df', util.iris(['class'], slice_size)]

    arguments = {
        'parameters': {'value': 'Iris-setosa',
                       'attributes': ['class']},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        instance = ReplaceValuesOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df': df[1]})
    assert "Parameter value and replacement must be informed if is using replace" \
           " by value in task" in str(val_err)


def test_replace_no_output_implies_no_code_success():
    slice_size = 10
    df = ['df', util.iris(['class'], slice_size)]

    arguments = {
        'parameters': {'value': 'Iris-setosa', 'replacement': 'replaced',
                       'attributes': ['class']},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {}
    }
    instance = ReplaceValuesOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})

    assert not instance.has_code
