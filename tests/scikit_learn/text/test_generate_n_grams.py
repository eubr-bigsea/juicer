from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import GenerateNGramsOperation
import pytest

# GenerateNGrams
# 
def test_generate_n_grams_success():
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
    instance = GenerateNGramsOperation(**arguments)
    result = util.execute(instance.generate_code(), 
                          dict([df]))
    assert result['out'].equals(util.iris(size=slice_size))
