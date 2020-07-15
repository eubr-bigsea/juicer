from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import AggregationOperation
import pandas as pd
import pytest


def _collect_list(x):
    return x.tolist()


def _collect_set(x):
    return set(x.tolist())


# Aggregation
# Test avg, collect_list, collect_set, count, first, last, max, min, sum, size
#


class IrisAggregationOperation:
    def __init__(self, slice_size=None, df=None, arguments=None, instance=None,
                 result=None):
        self.slice_size = 10 if slice_size is None else slice_size
        self.df = ['df', util.iris(['class', 'sepalwidth', 'petalwidth'],
                                   self.slice_size)] if df is None else df
        self.arguments = {
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: ['class'],
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': 'sepalwidth',
                      'aggregate': ['count'],
                      'alias': ['count_sepalwidth']}]
            },
            'named_inputs': {
                'input data': self.df[0],
            },
            'named_outputs': {
                'output data': 'out'
            }
        } if arguments is None else arguments
        self.instance = AggregationOperation(
            **self.arguments) if instance is None else instance
        self.result = util.execute(self.instance.generate_code(), {
            'df': self.df[1]}) if result is None else result


def test_aggregation_success():
    operation = IrisAggregationOperation()

    columns = ['class']
    target = {'sepalwidth': ['count_sepalwidth']}
    operations = {"sepalwidth": ['count']}
    out = operation.df[1].groupby(columns).agg(operations)

    new_idx = []
    old = None
    i = 0
    for (n1, n2) in out.columns:
        if old != n1:
            old = n1
            i = 0
        new_idx.append(target[n1][i])
        i += 1

    out.columns = new_idx
    out = out.reset_index()
    out.reset_index(drop=True, inplace=True)
    assert operation.result['out'].equals(out)


def test_multiple_aggregation_success():
    operation = IrisAggregationOperation(arguments={
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [{'attribute': 'sepalwidth',
                  'aggregate': ['avg', 'collect_list', 'collect_set', 'count',
                                'first', 'last', 'max', 'min', 'sum', 'size'],
                  'alias': ['sepal_avg', 'sepal_collect_list',
                            'sepal_collect_set',
                            'sepal_count', 'sepal_first', 'sepal_last',
                            'sepal_max', 'sepal_min', 'sepal_sum',
                            'sepal_size']}]
        },
        'named_inputs': {
            'input data': IrisAggregationOperation(arguments=False,
                                                   instance=False,
                                                   result=False).df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    })

    columns = ['class']
    target = {
        'sepalwidth': ['sepal_avg', 'sepal_collect_list', 'sepal_collect_set',
                       'sepal_count', 'sepal_first', 'sepal_last', 'sepal_max',
                       'sepal_min', 'sepal_sum', 'sepal_size']}
    operations = {
        "sepalwidth": ['mean', _collect_list, _collect_set, 'count', 'first',
                       'last', 'max', 'min', 'sum', 'size']}
    out = operation.df[1].groupby(columns).agg(operations)

    new_idx = []
    old = None
    i = 0
    for (n1, n2) in out.columns:
        if old != n1:
            old = n1
            i = 0
        new_idx.append(target[n1][i])
        i += 1

    out.columns = new_idx
    out = out.reset_index()
    out.reset_index(drop=True, inplace=True)
    assert operation.result['out'].equals(out)


def test_multiple_dicts_success():
    # Feature or Bug?

    # You can pass multiple dicts to FUNCTION_PARAM and this allows to
    # specify each 'attribute', 'aggregate' and 'alias'.
    # In the test below, 'sepalwidth' receives 'sum' and 'size' with their
    # respective aliases, and 'petalwidth' receives 'min' and 'max' also
    # with their own aliases.

    operation = IrisAggregationOperation(arguments={
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [{'attribute': 'sepalwidth',
                  'aggregate': ['sum', 'size'],
                  'alias': ['sepal_sum', 'sepal_size']},
                 {'attribute': 'petalwidth',
                  'aggregate': ['min', 'max'],
                  'alias': ['petal_min', 'petal_max']}]
        },
        'named_inputs': {
            'input data': IrisAggregationOperation(arguments=False,
                                                   instance=False,
                                                   result=False).df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    })

    columns = ['class']
    target = {'sepalwidth': ['sepal_sum', 'sepal_size'],
              'petalwidth': ['petal_min', 'petal_max']}
    operations = {"sepalwidth": ['sum', 'size'], "petalwidth": ['min', 'max']}
    out = operation.df[1].groupby(columns).agg(operations)

    new_idx = []
    old = None
    i = 0
    for (n1, n2) in out.columns:
        if old != n1:
            old = n1
            i = 0
        new_idx.append(target[n1][i])
        i += 1

    out.columns = new_idx
    out = out.reset_index()
    out.reset_index(drop=True, inplace=True)
    assert operation.result['out'].equals(out)


def test_aggregation_fail_missing_parameters():
    with pytest.raises(ValueError) as value_error:
        IrisAggregationOperation(arguments={
            'parameters': {},
            'named_inputs': {
                'input data': IrisAggregationOperation(
                    arguments=False,
                    instance=False,
                    result=False).df[0],
            },
            'named_outputs': {
                'output data': 'out'
            }
        })
    assert "Parameter 'function' must be informed for task" in str(value_error)


def test_aggregation_fail_missing_attributes_param():
    with pytest.raises(TypeError) as type_error:
        IrisAggregationOperation(arguments={
            'parameters': {
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': 'sepalwidth', 'alias': ['sum_sepalwidth'],
                      'aggregate': ['sum']}]},
            'named_inputs': {
                'input data': IrisAggregationOperation(arguments=False,
                                                       instance=False,
                                                       result=False).df[0]
            },
            'named_outputs': {
                'output data': 'out'
            }
        })
    assert "You have to supply one of 'by' and 'level'" in str(type_error)


def test_aggregation_fail_invalid_attributes_param():
    with pytest.raises(KeyError) as key_error:
        IrisAggregationOperation(arguments={
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: ['invalid'],
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': 'sepalwidth', 'alias': ['sum_sepalwidth'],
                      'aggregate': ['sum']}]},
            'named_inputs': {
                'input data': IrisAggregationOperation(arguments=False,
                                                       instance=False,
                                                       result=False).df[0]
            },
            'named_outputs': {
                'output data': 'out'
            }
        })
    assert "ExceptionInfo KeyError('invalid')" in str(key_error)


def test_aggregation_fail_invalid_function_param_attribute():
    with pytest.raises(KeyError) as key_error:
        IrisAggregationOperation(arguments={
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: ['class'],
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': 'invalid', 'alias': ['sum_invalid'],
                      'aggregate': ['sum']}]},
            'named_inputs': {
                'input data': IrisAggregationOperation(arguments=False,
                                                       instance=False,
                                                       result=False).df[0]
            },
            'named_outputs': {
                'output data': 'out'
            }
        })
    assert 'ExceptionInfo KeyError("Column \'invalid\' does not exist!")' in str(
        key_error)


def test_aggregation_fail_invalid_function_param_aggregate():
    with pytest.raises(KeyError) as key_error:
        IrisAggregationOperation(arguments={
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: ['class'],
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': 'class', 'alias': ['invalid_class'],
                      'aggregate': ['invalid']}]},
            'named_inputs': {
                'input data': IrisAggregationOperation(arguments=False,
                                                       instance=False,
                                                       result=False).df[0]
            },
            'named_outputs': {
                'output data': 'out'
            }
        }, result=False)
    assert "ExceptionInfo KeyError('invalid')" in str(key_error)


def test_aggregation_fail_missing_input_port():
    with pytest.raises(KeyError) as key_error:
        IrisAggregationOperation(arguments={
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: ['class'],
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': 'class', 'alias': ['total_per_class'],
                      'aggregate': ['count']}]},
            'named_inputs': {},
            'named_outputs': {
                'output data': 'out'
            }
        })
    assert "ExceptionInfo KeyError('input data')" in str(key_error)


def test_aggregation_success_no_output_implies_no_code():
    # Weird one, I don't know if it's really not generating code.

    operation = IrisAggregationOperation(arguments={
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [{'attribute': 'class', 'alias': ['total_per_class'],
                  'aggregate': ['count']}]},
        'named_inputs': {
            'input data': IrisAggregationOperation(arguments=False,
                                                   instance=False,
                                                   result=False).df[0]
        },
        'named_outputs': {}
    }, result=False)
    assert not operation.instance.has_code


def test_aggregation_success_non_numeric_attributes():
    operation = IrisAggregationOperation(
        df=['df', util.titanic(['name', 'homedest'],
                               IrisAggregationOperation(arguments=False,
                                                        instance=False,
                                                        result=False).slice_size)],
        arguments={
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: ['name', 'homedest'],
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': 'homedest',
                      'aggregate': ['size', 'sum', 'min', 'max', 'last', 'first',
                                    'count', 'collect_set', 'collect_list'],
                      'alias': ['homedest_size', 'homedest_sum', 'homedest_min',
                                'homedest_max', 'homedest_last',
                                'homedest_first',
                                'homedest_count', 'collect_set',
                                'collect_list']}]
            },
            'named_inputs': {
                'input data': IrisAggregationOperation(arguments=False,
                                                       instance=False,
                                                       result=False).df[0],
            },
            'named_outputs': {
                'output data': 'out'
            }
        })

    columns = ['name', 'homedest']
    target = {
        'homedest': ['homedest_size', 'homedest_sum', 'homedest_min',
                     'homedest_max', 'homedest_last', 'homedest_first',
                     'homedest_count', 'collect_set', 'collect_list']}
    operations = {
        "homedest": ['size', 'sum', 'min', 'max', 'last', 'first', 'count',
                     _collect_set, _collect_list]}
    out = operation.df[1].groupby(columns).agg(operations)
    new_idx = []
    old = None
    i = 0
    for (n1, n2) in out.columns:
        if old != n1:
            old = n1
            i = 0
        new_idx.append(target[n1][i])
        i += 1

    out.columns = new_idx
    out = out.reset_index()
    out.reset_index(drop=True, inplace=True)
    assert operation.result['out'].equals(out)


def test_aggregation_fail_non_numeric_attributes():
    with pytest.raises(pd.core.base.DataError) as data_error:
        operation = IrisAggregationOperation(
            df=['df', util.titanic(['name', 'homedest'],
                                   IrisAggregationOperation(arguments=False,
                                                            instance=False,
                                                            result=False).slice_size)],
            arguments={
                'parameters': {
                    AggregationOperation.ATTRIBUTES_PARAM: ['name', 'homedest'],
                    AggregationOperation.FUNCTION_PARAM:
                        [{'attribute': 'homedest',
                          'aggregate': ['avg'],
                          'alias': ['homedest_avg']}]
                    # avg doesn't work with non numeric attribute
                },
                'named_inputs': {
                    'input data': IrisAggregationOperation(arguments=False,
                                                           instance=False,
                                                           result=False).df[0],
                },
                'named_outputs': {
                    'output data': 'out'
                }
            })
    assert "ExceptionInfo DataError('No numeric types to aggregate')" in \
           str(data_error)


def test_aggregation_success_with_pivot_table():
    operation = IrisAggregationOperation(arguments={
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [{'attribute': 'petalwidth',
                  'aggregate': ['count']}, {'attribute': 'sepalwidth',
                                            'aggregate': ['count']}],
            AggregationOperation.PIVOT_ATTRIBUTE: ['sepalwidth', 'class'],
        },

        'named_inputs': {
            'input data': IrisAggregationOperation(arguments=False,
                                                   instance=False,
                                                   result=False).df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    })

    aggfunc = {"petalwidth": ['count'], "sepalwidth": ['count']}
    out = pd.pivot_table(operation.df[1], index=['class'],
                         values=['petalwidth', 'sepalwidth'],
                         columns=['sepalwidth', 'class'], aggfunc=aggfunc)

    out.reset_index(inplace=True)
    new_idx = [n[0] if n[1] == ''
               else "%s_%s_%s" % (n[0], n[1], n[2])
               for n in out.columns.ravel()]
    out = pd.DataFrame(out.to_records())
    out.reset_index(drop=True, inplace=True)
    out = out.drop(columns='index')
    out.columns = new_idx

    assert operation.result['out'].equals(out)


def test_aggregation_success_with_pivot_attribute_and_value_attribute():
    # TODO
    # This seems deprecated...
    operation = IrisAggregationOperation(arguments={
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [{'attribute': 'sepalwidth',
                  'aggregate': ['count']}],
            AggregationOperation.PIVOT_ATTRIBUTE: ['class'],
            AggregationOperation.PIVOT_VALUE_ATTRIBUTE: False
        },

        'named_inputs': {
            'input data': IrisAggregationOperation(arguments=False,
                                                   instance=False,
                                                   result=False).df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    })
    print(operation.result['out'])


def xtest_aggregation_fail_with_pivot_attribute_and_value_attribute():
    pass
    # TODO

# TESTS TODO:

# test swith pivot attribute and pivot value attribute
# use of asterisk (requires changes in operation)

# DONE:
# tests with non numeric camps
# missing parameters;
# missing ports;
# invalid function name; (valids are avg, collect_list,
# collect_set, count, first, last, max, min, sum, size)
# use of size versus count (requires study and changes in the operation code)
# tests of pivot table
