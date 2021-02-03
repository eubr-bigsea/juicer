from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import AggregationOperation
import pandas as pd
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


def _collect_list(x):
    return x.tolist()


def _collect_set(x):
    return set(x.tolist())


def return_funcs(attribute, drop=None, add=None):
    function_list = ['avg', 'collect_list', 'collect_set', 'count', 'first',
                     'last', 'max', 'min', 'sum', 'size']
    if drop is not None:
        function_list.remove(drop)
    if add is not None:
        function_list.append(add)
    len_attr = int(len(attribute) / 2)
    if len_attr <= 2: len_attr = 3
    built = [{'attribute': attribute, 'f': function,
              'alias': f'{attribute[:len_attr]}_{function}'} for function
             in function_list]
    return built


# Aggregation
# Test avg, collect_list, collect_set, count, first, last, max, min, sum, size
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_aggregation_success():
    df = util.iris(['class'], size=150)
    test_out = df.copy()
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function':
                [{'attribute': 'class',
                  'f': 'count',
                  'alias': 'class_count'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_out.groupby(['class']).agg(class_count=(
        'class', 'count')).reset_index()
    assert result['out'].equals(test_out)


def test_aggregation_asterisk_success():
    df = util.iris(['class'], size=150)
    test_out = df.copy()
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function':
                [{'attribute': '*',
                  'f': 'count',
                  'alias': 'class_count'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_out.groupby(['class']).agg(class_count=(
        'class', 'count')).reset_index()
    assert result['out'].equals(test_out)


def test_aggregation_multiple_functions_success():
    df = util.iris(['class', 'sepalwidth'], size=150)
    test_out = df.copy()
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function':
                return_funcs('sepalwidth')},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_out.groupby('class').agg(
        sepal_avg=('sepalwidth', 'mean'),
        sepal_collect_list=('sepalwidth', _collect_list),
        sepal_collect_set=('sepalwidth', _collect_set),
        sepal_count=('sepalwidth', 'count'),
        sepal_first=('sepalwidth', 'first'),
        sepal_last=('sepalwidth', 'last'),
        sepal_max=('sepalwidth', 'max'),
        sepal_min=('sepalwidth', 'min'),
        sepal_sum=('sepalwidth', 'sum'),
        sepal_size=('sepalwidth', 'size')).reset_index()
    assert result['out'].equals(test_out)


def test_aggregation_multiple_attributes_and_functions_success():
    """You can pass multiple dicts to FUNCTION_PARAM and this allows to
    specify each parameter ('attribute', 'f' and 'alias').
    In the test below, 'sepalwidth' receives 'sum' and 'size' with their
    respective aliases, and 'petalwidth' receives 'min' and 'max' also
    with their own aliases."""
    df = util.iris(['sepalwidth', 'petalwidth', 'class'], size=150)
    test_out = df.copy()
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function':
                [{'attribute': 'sepalwidth', 'f': 'sum', 'alias': 'sepal_sum'},
                 {'attribute': 'sepalwidth', 'f': 'size', 'alias': 'sepal_size'},
                 {'attribute': 'petalwidth', 'f': 'min', 'alias': 'petal_min'},
                 {'attribute': 'petalwidth', 'f': 'max', 'alias': 'petal_max'}
                 ]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_out.groupby(['class']).agg(
        sepal_sum=("sepalwidth", "sum"),
        sepal_size=("sepalwidth", "size"),
        petal_min=("petalwidth", "min"),
        petal_max=("petalwidth", "max")
    ).reset_index()
    assert result['out'].equals(test_out)


def test_aggregation_non_numeric_attributes_success():
    df = util.titanic(['homedest'], size=150)
    test_out = df.copy()

    arguments = {
        'parameters': {
            'attributes': ['homedest'],
            'function': return_funcs('homedest', drop='avg')
        },
        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    test_out = test_out.groupby(['homedest']).agg(
        home_collect_list=('homedest', _collect_list),
        home_collect_set=('homedest', _collect_set),
        home_count=('homedest', 'count'), home_first=('homedest', 'first'),
        home_last=('homedest', 'last'), home_max=('homedest', 'max'),
        home_min=('homedest', 'min'), home_sum=('homedest', 'sum'),
        home_size=('homedest', 'size')).reset_index()
    assert result['out'].equals(test_out)


def test_aggregation_pivot_table_success():
    df = util.iris(['class', 'sepalwidth', 'petalwidth'], size=150)
    test_out = df.copy()
    arguments = {
        'parameters': {
            'attributes': ['petalwidth'],
            'function':
                [{'attribute': 'petalwidth', 'f': 'count'}],
            'pivot': ['class'],
        },

        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    aggfunc = {"petalwidth": ['count']}
    test_out = pd.pivot_table(test_out, index=['petalwidth'],
                              columns=['class'], aggfunc=aggfunc)
    test_out.reset_index(inplace=True)
    new_idx = [n[0] if n[1] == ''
               else "%s_%s_%s" % (n[0], n[1], n[2])
               for n in test_out.columns]
    test_out.columns = new_idx
    assert result['out'].equals(test_out)


def test_aggregation_pivot_attribute_param_and_value_attribute_param_success():
    df = util.iris(['class', 'sepalwidth', 'petalwidth'], size=10)
    test_out = df.copy()
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function':
                [{'attribute': 'class', 'f': 'count'}],
            'pivot': ['class'],
            'pivot_values': '"Iris-setosa"'
        },

        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    aggfunc = {"class": ['count']}
    values = ["Iris-setosa"]
    input_data = test_out.loc[test_out['class'].isin(values)]
    test_out = pd.pivot_table(input_data, index=['class'],
                              columns=['class'], aggfunc=aggfunc)
    test_out.reset_index(inplace=True)
    new_idx = [n[0] if n[1] == ''
               else "%s_%s_%s" % (n[0], n[1], n[2])
               for n in test_out.columns]
    test_out.columns = new_idx
    assert result['out'].equals(test_out)


def test_aggregation_no_output_implies_no_code_success():
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function':
                [{'attribute': 'class',
                  'f': 'count',
                  'alias': 'class_count'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = AggregationOperation(**arguments)
    assert instance.generate_code() is None


def test_aggregation_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function':
                [{'attribute': 'class',
                  'f': 'count',
                  'alias': 'class_count'}]},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_aggregation_missing_attribute_param_fail():
    df = util.iris(['class'], size=150)
    arguments = {
        'parameters': {
            'function':
                [{'attribute': 'class',
                  'f': 'count',
                  'alias': 'class_count'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    with pytest.raises(TypeError) as typ_err:
        util.execute(util.get_complete_code(instance),
                     {'df': df})
    assert "You have to supply one of 'by' and 'level'" in str(typ_err.value)


def test_aggregation_missing_function_param_fail():
    arguments = {
        'parameters': {
            'attributes': ['class']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        AggregationOperation(**arguments)
    assert "Parameter 'function' must be informed for task" in str(
        val_err.value)


def test_aggregation_missing_function_param_attribute_fail():
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function': [{'f': 'count',
                          'alias': 'class_count'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    with pytest.raises(KeyError) as key_err:
        AggregationOperation(**arguments)
    assert "attribute" in str(key_err.value)


def test_aggregation_missing_function_param_function_fail():
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function': [{'attribute': 'class',
                          'alias': 'class_count'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(KeyError) as key_err:
        AggregationOperation(**arguments)
    assert "f" in str(key_err.value)


def test_aggregation_missing_function_param_alias_fail():
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function': [{'attribute': 'class',
                          'f': 'count'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    with pytest.raises(KeyError) as key_err:
        AggregationOperation(**arguments)
    assert "alias" in str(key_err.value)


def test_aggregation_invalid_attribute_param_fail():
    df = util.iris(['class'], size=150)
    arguments = {
        'parameters': {
            'attributes': 'invalid',
            'function':
                [{'attribute': 'class',
                  'f': 'count',
                  'alias': 'class_count'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    with pytest.raises(NameError) as nam_err:
        util.execute(util.get_complete_code(instance),
                     {'df': df})
    assert "name 'invalid' is not defined" in str(nam_err.value)


def test_aggregation_invalid_function_param_fail():
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function': 'invalid'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(TypeError) as typ_err:
        AggregationOperation(**arguments)
    assert "string indices must be integers" in str(typ_err.value)


def test_aggregation_invalid_function_param_attribute_fail():
    df = util.iris(['class'], size=150)
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function':
                [{'attribute': 'invalid',
                  'f': 'count',
                  'alias': 'class_count'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(util.get_complete_code(instance),
                     {'df': df})
    assert "Column 'invalid' does not exist!" in str(key_err.value)


def test_aggregation_invalid_function_param_function_fail():
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function':
                [{'attribute': 'class',
                  'f': 'invalid',
                  'alias': 'class_count'}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(KeyError) as key_err:
        AggregationOperation(**arguments)
    assert 'invalid' in str(key_err.value)


def test_aggregation_invalid_function_param_alias_fail():
    df = util.iris(['class'], size=150)
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function':
                [{'attribute': 'class',
                  'f': 'count',
                  'alias': ''}]},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    with pytest.raises(SyntaxError) as syn_err:
        util.execute(util.get_complete_code(instance),
                     {'df': df})
    assert "invalid syntax" in str(syn_err.value)


def test_aggregation_invalid_pivot_table_fail():
    df = util.iris(['class', 'sepalwidth', 'petalwidth'], size=150)
    arguments = {
        'parameters': {
            'attributes': ['petalwidth'],
            'function':
                [{'attribute': 'petalwidth', 'f': 'count'}],
            'pivot': 'invalid',
        },

        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    with pytest.raises(NameError) as nam_err:
        util.execute(util.get_complete_code(instance),
                     {'df': df})
    assert "name 'invalid' is not defined" in str(nam_err.value)


def test_aggregation_invalid_pivot_value_attribute_fail():
    df = util.iris(['class', 'sepalwidth', 'petalwidth'], size=10)
    arguments = {
        'parameters': {
            'attributes': ['class'],
            'function':
                [{'attribute': 'class', 'f': 'count'}],
            'pivot': ['class'],
            'pivot_values': 'invalid'
        },

        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    with pytest.raises(NameError) as nam_err:
        util.execute(util.get_complete_code(instance),
                     {'df': df})
    assert "name 'invalid' is not defined" in str(nam_err.value)


def test_aggregation_non_numeric_attributes_fail():
    df = util.titanic(['homedest'], size=150)
    arguments = {
        'parameters': {
            'attributes': ['homedest'],
            'function': return_funcs('homedest')
        },
        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AggregationOperation(**arguments)
    with pytest.raises(pd.core.base.DataError) as data_err:
        util.execute(util.get_complete_code(instance),
                     {'df': df})
    assert "No numeric types to aggregate" in str(data_err.value)

# DONE:
# tests with non numeric camps
# missing parameters;
# missing ports;
# invalid function name; (valids are avg, collect_list,
# collect_set, count, first, last, max, min, sum, size)
# use of size versus count (requires study and changes in the operation code)
# tests with pivot table
# use of asterisk (requires changes in operation)
# tests with pivot attribute and pivot value attribute
