from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import AggregationOperation
import pandas as pd
import pytest


# Aggregation
# Test avg, collect_list, collect_set, count, first, last, max, min, sum
#

class IrisAggregationOperationDebug:
    def __init__(self, slice_size=None, df=None, arguments=None, instance=None, result=None):
        self.slice_size = 10 if slice_size is None else slice_size
        self.df = ['df', util.iris(['class'], slice_size)] if df is None else df
        self.arguments = {
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: ['class'],
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': '*', 'alias': 'total_per_class', 'f': 'count'}]},
            'named_inputs': {
                'input data': self.df[0],
            },
            'named_outputs': {
                'output data': 'out'
            }
        } if arguments is None else arguments
        self.instance = AggregationOperation(**self.arguments) if instance is None else instance
        self.result = util.execute(self.instance.generate_code(), {'df': self.df[1]}) if result is None else result


def xtest_aggregation_count_with_asterisc_success():
    """ 
    Count is the only aggregation function that allows to use '*'.
    It is not working with scikit implementation.
    """
    operation = IrisAggregationOperationDebug()
    print('=' * 10)
    print(operation.instance.generate_code())
    print('=' * 10)
    result = util.execute(operation.instance.generate_code(), {'df': operation.df[1]})


def xtest_aggregation_fail_missing_parameters():
    with pytest.raises(ValueError) as value_error:
        operation = IrisAggregationOperationDebug(arguments={
            'parameters': {},
            'named_inputs': {
                'input data': IrisAggregationOperationDebug().df[0],
            },
            'named_outputs': {
                'output data': 'out'
            }
        })
    print(value_error)


def xtest_aggregation_fail_invalid_parameters():
    with pytest.raises(KeyError) as key_error:
        operation = IrisAggregationOperationDebug(arguments={
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: ['invalid'],
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': 'invalid', 'alias': 'invalid', 'f': 'invalid'}]},
            'named_inputs': {
                'input data': IrisAggregationOperationDebug().df[0]
            },
            'named_outputs': {
                'output data': 'out'
            }
        })
    print(key_error)


def xtest_aggregation_fail_blank_parameters():
    with pytest.raises(ValueError) as value_error:
        operation = IrisAggregationOperationDebug(arguments={
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: [''],
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': '', 'alias': '', 'f': ''}]},
            'named_inputs': {
                'input data': IrisAggregationOperationDebug().df[0],
            },
            'named_outputs': {
                'output data': 'out'
            }
        })
    print(value_error)


def xtest_aggregation_fail_missing_input_port():
    with pytest.raises(KeyError) as key_error:
        operation = IrisAggregationOperationDebug(arguments={
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: ['class'],
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': '*', 'alias': 'total_per_class', 'f': 'count'}]},
            'named_inputs': {},
            'named_outputs': {
                'output data': 'out'
            }
        })
    print(key_error)


def xtest_aggregation_success_no_output_implies_no_code():
    operation = IrisAggregationOperationDebug(arguments={
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [{'attribute': '*', 'alias': 'total_per_class', 'f': 'count'}]},
        'named_inputs': {
            'input data': IrisAggregationOperationDebug().df[0]
        },
        'named_outputs': {}
    })
    assert not operation.instance.has_code


class IrisAggregationOperation:
    def __init__(self, slice_size=None, iris=None, df=None, species=None, functions=None,
                 arguments=None, instance=None, result=None):

        self.slice_size = 10 if slice_size is None else slice_size

        self.iris = util.iris(['sepalwidth', 'class'], self.slice_size) if iris is None else iris

        self.df = ['df', self.iris] if df is None else df

        self.species = self.iris['class'].tolist() if species is None else species

        self.functions = {
            'avg': lambda x: 1.0 * sum(x) / len(x),
            'count': lambda x: len(x),
            'first': lambda x: x[0],
            'last': lambda x: x[-1],
            'collect_list': lambda x: list(x),
            'collect_set': lambda x: set(list(x)),
            'max': lambda x: max(x),
            'min': lambda x: min(x),
            'sum': lambda x: sum(x),
        } if functions is None else functions

        self.result_by_specie = {}
        for specie in self.species:
            values = self.iris[self.iris['class'] == specie]['sepalwidth'].tolist()
            self.result_by_specie[specie] = {}
            for k, function in self.functions.items():
                self.result_by_specie[specie][k] = function(values)

        self.arguments = {
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: ['class'],
                AggregationOperation.FUNCTION_PARAM:
                    [
                        {'attribute': 'sepalwidth', 'alias': k, 'f': k}
                        for k in self.functions.keys()
                    ]},
            'named_inputs': {
                'input data': self.df[0],
            },
            'named_outputs': {
                'output data': 'out'
            }
        } if arguments is None else arguments

        self.instance = AggregationOperation(**self.arguments) if instance is None else instance

        self.result = util.execute(self.instance.generate_code(), {'df': self.df[1]}) if result is None else result


def test_aggregation_by_class_success():
    operation = IrisAggregationOperation()
    for inx, val in operation.result['out'].iterrows():
        specie = val['class']
        for k in operation.functions.keys():
            assert pytest.approx(
                operation.result_by_specie[specie][k], 0.00001) == val[k]


def test_aggregation_by_class_fail_invalid_function_name():
    with pytest.raises(KeyError) as key_error:
        operation = IrisAggregationOperation(functions={'invalid': lambda x: sum(x)})
    print(key_error)


def test_aggregation_by_class_fail_missing_parameters():
    operation = IrisAggregationOperation()
    with pytest.raises(ValueError) as value_error:
        operation = IrisAggregationOperation(arguments={
            'parameters': {},
            'named_inputs': {'input data': operation.df[0]},
            'named_outputs': {'output data': 'out'}
        })
    print(value_error)


def test_aggregation_by_class_fail_no_function_passed():
    with pytest.raises(ValueError) as value_error:
        operation = IrisAggregationOperation(functions={})
    print(value_error)


def test_aggregation_by_class_fail_missing_input_port():
    operation = IrisAggregationOperation()
    with pytest.raises(KeyError) as key_error:
        operation = IrisAggregationOperation(arguments={'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [
                    {'attribute': 'sepalwidth', 'alias': k, 'f': k}
                    for k in operation.functions.keys()
                ]},
            'named_inputs': {},
            'named_outputs': {'output data': 'out'}
        })
    print(key_error)


def test_aggregation_by_class_success_no_output_implies_no_code():
    operation = IrisAggregationOperation()
    operation = IrisAggregationOperation(arguments={'parameters': {
        AggregationOperation.ATTRIBUTES_PARAM: ['class'],
        AggregationOperation.FUNCTION_PARAM:
            [
                {'attribute': 'sepalwidth', 'alias': k, 'f': k}
                for k in operation.functions.keys()
            ]},
        'named_inputs': {'input data': operation.df[0]},
        'named_outputs': {}
    })

    assert not operation.instance.has_code


def test_aggregation_by_class_success_with_pivot_table():
    # TODO:
    # Needs an assertion
    operation = IrisAggregationOperation(iris=util.iris(['sepalwidth', 'class'], 50),
                                         functions={
                                             'count': lambda x: len(x)})
    operation = IrisAggregationOperation(arguments={
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [
                    {'attribute': 'sepalwidth', 'alias': k, 'f': k}
                    for k in operation.functions.keys()
                ],
            AggregationOperation.PIVOT_ATTRIBUTE: ['class']},
        'named_inputs': {
            'input data': IrisAggregationOperation().df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    })

    print(f'{operation.result["out"]}')


def test_aggregation_by_class_success_non_numeric_camps():
    slice_size = 10
    titanic = util.titanic(['name', 'homedest'], slice_size)
    df = ['df', titanic]

    homedests = titanic['homedest'].tolist()
    functions = {
        'count': lambda x: len(x),
        'first': lambda x: x[0],
        'last': lambda x: x[-1],
        'collect_list': lambda x: list(x),
        'collect_set': lambda x: set(list(x)),
        'max': lambda x: max(x),
        'min': lambda x: min(x),
    }
    result_by_homedest = {}
    for homedest in homedests:
        values = titanic[titanic['homedest'] == homedest]['name'].tolist()
        result_by_homedest[homedest] = {}
        for k, function in functions.items():
            result_by_homedest[homedest][k] = function(values)

    arguments = {
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['homedest'],
            AggregationOperation.FUNCTION_PARAM:
                [
                    {'attribute': 'name', 'alias': k, 'f': k}
                    for k in functions.keys()
                ]},
        'named_inputs': {'input data': df[0]},
        'named_outputs': {'output data': 'out'}
    }
    instance = AggregationOperation(**arguments)
    result = util.execute(instance.generate_code(), {'df': df[1]})

    for inx, val in result['out'].iterrows():
        homedest = val['homedest']
        for k in functions.keys():
            assert result_by_homedest[homedest][k] == val[k]


def test_aggregation_by_class_fail_non_numeric_camps():
    slice_size = 10
    titanic = util.titanic(['name', 'homedest'], slice_size)
    df = ['df', titanic]

    homedests = titanic['homedest'].tolist()
    functions = {
        'avg': lambda x: 1.0 * sum(x) / len(x),  # avg doesn't work with non numeric
        'sum': lambda x: sum(x),  # sum doesn't work with non numeric
    }
    result_by_homedest = {}
    for homedest in homedests:
        values = titanic[titanic['homedest'] == homedest]['name'].tolist()
        result_by_homedest[homedest] = {}

        with pytest.raises(TypeError) as type_error:
            for k, function in functions.items():
                result_by_homedest[homedest][k] = function(values)
    print(type_error)

    arguments = {
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['homedest'],
            AggregationOperation.FUNCTION_PARAM:
                [
                    {'attribute': 'name', 'alias': k, 'f': k}
                    for k in functions.keys()
                ]},
        'named_inputs': {'input data': df[0]},
        'named_outputs': {'output data': 'out'}
    }

    instance = AggregationOperation(**arguments)

# Tests
# TODO:
# test of pivot table;
# use of size versus count (requires study and changes in the operation code);
# use of asterisk (requires changes in operation);

# OK:
# missing parameters;
# missing ports;
# invalid function name; (valids are avg, collect_list, collect_set, count, first, last, max, min, sum)
# tests with non numeric camps;
