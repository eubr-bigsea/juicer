from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import LSHOperation
import pytest
# LSH
# 
def test_lsh_success():
    slice_size = 10
    df = ['df', util.iris(['sepallength', 'sepalwidth', 
        'petalwidth', 'petallength'], slice_size)]

    arguments = {
        'parameters': {'label': 'species'},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LSHOperation(**arguments)
    with pytest.raises(ValueError) as val_err:
        util.execute(instance.generate_code(), dict([df]))
    assert f"Deprecated in Scikit-Learn" \
           in str(val_err.value)
    # assert result['out'].equals(util.iris(size=slice_size))
