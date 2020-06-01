from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import AggregationOperation
import pandas as pd
import pytest


# Aggregation
#

def test_aggregation_success():
    slice_size = 10
    df = ['df', util.iris('sepallength', slice_size)]

    arguments = {

        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: 'sepallenght',
            AggregationOperation.FUNCTION_PARAM:
                [{'attribute': 'sepallenght', 'alias': 'media_sepallenght', 'f': 'avg'}]},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    instance = AggregationOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df[1]})
    assert result['out'].equals(util.iris(size=slice_size))
