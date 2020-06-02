from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import AggregationOperation
import pandas as pd
import pytest


# Aggregation
# Test avg, collect_list, collect_set, count, first, last, max, min, sum
#

def xtest_aggregation_count_with_asterisc_success():
    """ 
    Count is the only aggregation function that allows to use '*'.
    It is not working with scikit implementation.
    """
    slice_size = 10
    df = ['df', util.iris(['class'], slice_size)]

    arguments = {
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [{'attribute': '*', 'alias': 'total_per_class', 'f': 'count'}]},
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    print('=' * 10)
    print(instance.generate_code())
    print('=' * 10)

    result = util.execute(instance.generate_code(), {'df': df[1]})

def test_aggregation_by_class_success():
    iris = util.iris(['sepalwidth', 'class'])
    df = ['df', iris]

    species = iris['class'].tolist()
    functions = {
            'avg': lambda x: 1.0 * sum(x) / len(x), 
            'count': lambda x: len(x),
            'first': lambda x: x[0],
            'last': lambda x: x[-1],
            'collect_list': lambda x: list(x),
            'collect_set':lambda x: set(list(x)),
            'max': lambda x: max(x),
            'min': lambda x: min(x),
            'sum': lambda x: sum(x),
    }
    result_by_specie = {}
    for specie in species:
        values = iris[iris['class'] == specie]['sepalwidth'].tolist()
        result_by_specie[specie] = {}
        for k, function in functions.items():
            result_by_specie[specie][k] = function(values)
    
    arguments = {
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [
                    {'attribute': 'sepalwidth', 'alias': k, 'f': k}
                    for k in functions.keys()
                ]},
        'named_inputs': { 'input data': df[0], },
        'named_outputs': { 'output data': 'out' }
    }
    instance = AggregationOperation(**arguments)

    result = util.execute(instance.generate_code(), {'df': df[1]})

    for inx, val in result['out'].iterrows():
        specie = val['class']
        for k in functions.keys():
            assert pytest.approx(
                    result_by_specie[specie][k], 0.00001) == val[k]

# TODO: 
# Tests: missing parameters, missing ports, invalid function name (valid are 
# avg, collect_list, collect_set, count, first, last, max, min, sum), 
# use of size versus count (requires study and changes in the operation code),
# Use of asterisk (requires changes in operation),
# Test of pivot table

