from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SGDRegressor
import pytest

# SGDRegressor:
# 
def test_sgd_regressor_success():
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
    instance = SGDRegressor(**arguments)
    result = util.execute(instance.generate_code(), 
                          dict([df]))
    assert result['out'].equals(util.iris(size=slice_size))
