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
                      'f': 'count',
                      'alias': 'count_sepalwidth'}]
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
    out = operation.df[1]\
        .groupby(['class'])\
        .agg(count_sepalwidth=('sepalwidth', 'count')).reset_index()
    assert operation.result['out'].equals(out)


def test_multiple_aggregation_success():
    operation = IrisAggregationOperation(arguments={
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [{'attribute': 'sepalwidth',
                  'f': 'avg', 'alias': 'sepal_avg'},
                 {'attribute': 'sepalwidth',
                  'f': 'collect_list', 'alias': 'sepal_collect_list'},
                 {'attribute': 'sepalwidth',
                  'f': 'collect_set', 'alias': 'sepal_collect_set'},
                 {'attribute': 'sepalwidth',
                  'f': 'count', 'alias': 'sepal_count'},
                 {'attribute': 'sepalwidth',
                  'f': 'first', 'alias': 'sepal_first'},
                 {'attribute': 'sepalwidth',
                  'f': 'last', 'alias': 'sepal_last'},
                 {'attribute': 'sepalwidth',
                  'f': 'max', 'alias': 'sepal_max'},
                 {'attribute': 'sepalwidth',
                  'f': 'min', 'alias': 'sepal_min'},
                 {'attribute': 'sepalwidth',
                  'f': 'sum', 'alias': 'sepal_sum'},
                 {'attribute': 'sepalwidth',
                  'f': 'size', 'alias': 'sepal_size'}]
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

    out = operation.df[1].groupby(columns)\
        .agg(sepal_avg=('sepalwidth', 'mean'),
             sepal_collect_list=('sepalwidth', _collect_list),
             sepal_collect_set=('sepalwidth', _collect_set),
             sepal_count=('sepalwidth', 'count'),
             sepal_first=('sepalwidth', 'first'),
             sepal_last=('sepalwidth', 'last'),
             sepal_max=('sepalwidth', 'max'),
             sepal_min=('sepalwidth', 'min'),
             sepal_sum=('sepalwidth', 'sum'),
             sepal_size=('sepalwidth', 'size'))\
        .reset_index()

    assert operation.result['out'].equals(out)


def test_multiple_dicts_success():
    # You can pass multiple dicts to FUNCTION_PARAM and this allows to
    # specify each 'attribute', 'f' and 'alias'.
    # In the test below, 'sepalwidth' receives 'sum' and 'size' with their
    # respective aliases, and 'petalwidth' receives 'min' and 'max' also
    # with their own aliases.

    operation = IrisAggregationOperation(arguments={
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [{'attribute': 'sepalwidth', 'f': 'sum', 'alias': 'sepal_sum'},
                 {'attribute': 'sepalwidth', 'f': 'size',
                  'alias': 'sepal_size'},
                 {'attribute': 'petalwidth', 'f': 'min', 'alias': 'petal_min'},
                 {'attribute': 'petalwidth', 'f': 'max', 'alias': 'petal_max'}
                 ]
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

    out = operation.df[1].groupby(['class'])\
        .agg(sepal_sum=("sepalwidth", "sum"),
             sepal_size=("sepalwidth", "size"),
             petal_min=("petalwidth", "min"),
             petal_max=("petalwidth", "max")
             ).reset_index()

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
                    [{'attribute': 'sepalwidth', 'alias': 'sum_sepalwidth',
                      'f': 'sum'}]},
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
                    [{'attribute': 'sepalwidth', 'alias': 'sum_sepalwidth',
                      'f': 'sum'}]},
            'named_inputs': {
                'input data': IrisAggregationOperation(arguments=False,
                                                       instance=False,
                                                       result=False).df[0]
            },
            'named_outputs': {
                'output data': 'out'
            }
        })
    assert "ExceptionInfo KeyError('invalid'" in str(key_error)


def test_aggregation_fail_invalid_function_param_attribute():
    with pytest.raises(KeyError) as key_error:
        IrisAggregationOperation(arguments={
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: ['class'],
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': 'invalid', 'alias': 'sum_invalid',
                      'f': 'sum'}]},
            'named_inputs': {
                'input data': IrisAggregationOperation(arguments=False,
                                                       instance=False,
                                                       result=False).df[0]
            },
            'named_outputs': {
                'output data': 'out'
            }
        })
    assert 'ExceptionInfo KeyError("Column \'invalid\' does not exist!"' in \
           str(key_error)


def test_aggregation_fail_invalid_function_param_aggregate():
    with pytest.raises(KeyError) as key_error:
        IrisAggregationOperation(arguments={
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: ['class'],
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': 'class', 'alias': 'invalid_class',
                      'f': 'invalid'}]},
            'named_inputs': {
                'input data': IrisAggregationOperation(arguments=False,
                                                       instance=False,
                                                       result=False).df[0]
            },
            'named_outputs': {
                'output data': 'out'
            }
        }, result=False)
    assert "ExceptionInfo KeyError('invalid'" in str(key_error)


def test_aggregation_fail_missing_input_port():
    with pytest.raises(KeyError) as key_error:
        IrisAggregationOperation(arguments={
            'parameters': {
                AggregationOperation.ATTRIBUTES_PARAM: ['class'],
                AggregationOperation.FUNCTION_PARAM:
                    [{'attribute': 'class', 'alias': 'total_per_class',
                      'f': 'count'}]},
            'named_inputs': {},
            'named_outputs': {
                'output data': 'out'
            }
        })
    assert "ExceptionInfo KeyError('input data'" in str(key_error)


def test_aggregation_success_no_output_implies_no_code():

    operation = IrisAggregationOperation(arguments={
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [{'attribute': 'class', 'alias': 'total_per_class',
                  'f': 'count'}]},
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
                    [
                        {'attribute': 'homedest', 'f': 'size',
                         'alias': 'homedest_size'},
                        {'attribute': 'homedest', 'f': 'sum',
                         'alias': 'homedest_sum'},
                        {'attribute': 'homedest', 'f': 'min',
                         'alias': 'homedest_min'},
                        {'attribute': 'homedest', 'f': 'max',
                         'alias': 'homedest_max'},
                        {'attribute': 'homedest', 'f': 'last',
                         'alias': 'homedest_last'},
                        {'attribute': 'homedest', 'f': 'first',
                         'alias': 'homedest_first'},
                        {'attribute': 'homedest', 'f': 'count',
                         'alias': 'homedest_count'},
                        {'attribute': 'homedest', 'f': 'collect_set',
                         'alias': 'collect_set'},
                        {'attribute': 'homedest', 'f': 'collect_list',
                         'alias': 'collect_list'}
                    ]
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

    out = operation.df[1].groupby(['name', 'homedest'])\
        .agg(homedest_size=('homedest', 'size'),
             homedest_sum=('homedest', 'sum'),
             homedest_min=('homedest', 'min'),
             homedest_max=('homedest', 'max'),
             homedest_last=('homedest', 'last'),
             homedest_first=('homedest', 'first'),
             homedest_count=('homedest', 'count'),
             collect_set=('homedest', _collect_set),
             collect_list=('homedest', _collect_list),
             ).reset_index()

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
                          'f': 'avg',
                          'alias': 'homedest_avg'}]
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
    assert "ExceptionInfo DataError('No numeric types to aggregate'" \
           in str(data_error)


def test_aggregation_success_with_pivot_table():
    operation = IrisAggregationOperation(arguments={
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [{'attribute': 'petalwidth', 'f': 'count'},
                 {'attribute': 'sepalwidth', 'f': 'count'}],
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
                         columns=['sepalwidth', 'class'], aggfunc=aggfunc)
    # rename columns and convert to DataFrame
    out.reset_index(inplace=True)
    new_idx = [n[0] if n[1] == ''
               else "%s_%s_%s" % (n[0], n[1], n[2])
               for n in out.columns.ravel()]
    out.columns = new_idx

    assert operation.result['out'].equals(out)


def test_aggregation_success_with_pivot_attribute_and_value_attribute():

    operation = IrisAggregationOperation(arguments={
        'parameters': {
            AggregationOperation.ATTRIBUTES_PARAM: ['class'],
            AggregationOperation.FUNCTION_PARAM:
                [{'attribute': 'sepalwidth', 'f': 'count'}],
            AggregationOperation.PIVOT_ATTRIBUTE: ['class'],
            AggregationOperation.PIVOT_VALUE_ATTRIBUTE: '"Iris-setosa"'
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

    aggfunc = {"sepalwidth": ['count']}
    values = ["Iris-setosa"]
    input_data = operation.df[1].loc[operation.df[1]['class'].isin(values)]
    out = pd.pivot_table(input_data, index=['class'],
                         columns=['class'], aggfunc=aggfunc)
    # rename columns and convert to DataFrame
    out.reset_index(inplace=True)
    new_idx = [n[0] if n[1] == ''
               else "%s_%s_%s" % (n[0], n[1], n[2])
               for n in out.columns.ravel()]
    out.columns = new_idx

    assert operation.result['out'].equals(out)

# TESTS

# DONE:
# tests with non numeric camps
# missing parameters;
# missing ports;
# invalid function name; (valids are avg, collect_list,
# collect_set, count, first, last, max, min, sum, size)
# use of size versus count (requires study and changes in the operation code)
# tests of pivot table
# use of asterisk (requires changes in operation)
# test swith pivot attribute and pivot value attribute

