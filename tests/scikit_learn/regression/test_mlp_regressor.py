from tests.scikit_learn import util
from juicer.scikit_learn.regression_operation import MLPRegressorOperation
import pytest

# MLPRegressor
#

# TODO: This test will not work


def test_mlp_regressor_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth', 
        'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          dict([df]))
    assert result['out'].equals(util.iris(size=slice_size))
