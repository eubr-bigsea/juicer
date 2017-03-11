# -*- coding: utf-8 -*-
import ast
import json
from textwrap import dedent

import pytest
from juicer.spark.etl_operation import RandomSplit, Sort, Distinct, \
    SampleOrPartition, AddRows, Intersection, Difference, Join, Drop, \
    Transformation, Select, Aggregation, Filter, CleanMissing, AddColumns
from tests import compare_ast, format_code_comparison


def debug_ast(code, expected_code):
    print
    print code
    print '*' * 20
    print expected_code
    print '*' * 20


def test_add_columns_minimum_params_success():
    params = {}
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    instance = AddColumns(params, inputs, outputs,
                          named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = dedent("""
    w_{in0}_{in1} = Window().orderBy()
    {in0}_inx = {in0}.withColumn('_inx', rowNumber().over(w_{in0}_{in1}))
    {in1}_inx = {in1}.withColumn('_inx', rowNumber().over(w_{in0}_{in1}))

    {out} = {in0}_inx.join({in1}_inx, {in0}_inx._inx == {in1}_inx._inx,
                'inner').drop({in0}_inx._inx).drop({in1}_inx._inx)""".format(
        out=outputs[0], in0=inputs[0], in1=inputs[1]))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_add_rows_minimal_params_success():
    params = {}
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    instance = AddRows(params, inputs, outputs,
                       named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = "{out} = {in0}.unionAll({in1})".format(
        out=outputs[0], in0=inputs[0], in1=inputs[1])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_aggregation_rows_minimal_params_success():
    params = {
        Aggregation.FUNCTION_PARAM: [
            {'attribute': 'income', 'f': 'AVG', 'alias': 'avg_income'}],
        Aggregation.ATTRIBUTES_PARAM: ['country']
    }
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    instance = Aggregation(params, inputs, outputs,
                           named_inputs={}, named_outputs={})
    code = instance.generate_code()

    expected_code = """{out} = {in0}.groupBy(functions.col('{agg}'))\
                        .agg(functions.avg('income').alias('avg_income'))""".format(
        out=outputs[0], in0=inputs[0], in1=inputs[1], agg='country', )

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_aggregation_rows_group_all_missing_attributes_success():
    params = {
        Aggregation.FUNCTION_PARAM: [
            {'attribute': 'income', 'f': 'AVG', 'alias': 'avg_income'}],
    }
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    instance = Aggregation(params, inputs, outputs,
                           named_inputs={}, named_outputs={})
    code = instance.generate_code()

    expected_code = """{out} = {in0}.agg(
                        functions.avg('income').alias('avg_income'))""".format(
        out=outputs[0], in0=inputs[0], in1=inputs[1], agg='country', )
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_aggregation_missing_function_param_failure():
    params = {
        Aggregation.ATTRIBUTES_PARAM: ['country']
    }
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    with pytest.raises(ValueError):
        Aggregation(params, inputs, outputs,
                    named_inputs={}, named_outputs={})


def test_clean_missing_minimal_params_success():
    params = {
        CleanMissing.ATTRIBUTES_PARAM: ['name'],
        CleanMissing.MIN_MISSING_RATIO_PARAM: "0.0",
        CleanMissing.MAX_MISSING_RATIO_PARAM: "1.0",
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = CleanMissing(params, inputs, outputs,
                            named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = dedent("""
    ratio_{input_1} = {input_1}.select(
        (functions.count('{attribute}') / functions.count('*')).alias(
        '{attribute}')).collect()
    attributes_{input_1} = [c for c in ["{attribute}"]
                 if 0.0 <= ratio_{input_1}[0][c] <= 1.0]
    if len(attributes_input_1) > 0:
        {output_1} = {input_1}.na.drop(how='any', subset=attributes_{input_1})
    else:
        {output_1} = {input_1}
    """.format(input_1=inputs[0], attribute=params['attributes'][0],
               output_1=outputs[0]))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_clean_missing_without_missing_rating_params_success():
    params = {
        CleanMissing.ATTRIBUTES_PARAM: ['name'],
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = CleanMissing(params, inputs, outputs,
                            named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = dedent("""
    attributes_{input_1} = ['{attribute}']
    if len(attributes_input_1) > 0:
        {output_1} = {input_1}.na.drop(how='any', subset=attributes_{input_1})
    else:
        {output_1} = {input_1}
    """.format(input_1=inputs[0], attribute=params['attributes'][0],
               output_1=outputs[0]))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_clean_missing_minimal_params_type_value_success():
    params = {
        CleanMissing.ATTRIBUTES_PARAM: ['name'],
        CleanMissing.MIN_MISSING_RATIO_PARAM: "0.0",
        CleanMissing.MAX_MISSING_RATIO_PARAM: "1.0",
        CleanMissing.VALUE_PARAMETER: "200",
        CleanMissing.CLEANING_MODE_PARAM: CleanMissing.VALUE
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = CleanMissing(params, inputs, outputs,
                            named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = dedent("""
    ratio_{input_1} = {input_1}.select(
        (functions.count('{attribute}') / functions.count('*')).alias(
        '{attribute}')).collect()
    attributes_{input_1} = [c for c in ["{attribute}"]
                 if 0.0 <= ratio_{input_1}[0][c] <= 1.0]
    if len(attributes_input_1) > 0:
        {output_1} = {input_1}.na.fill(value={value},
                subset=attributes_{input_1})
    else:
        {output_1} = {input_1}
    """.format(input_1=inputs[0], attribute=params['attributes'][0],
               output_1=outputs[0], value=params[CleanMissing.VALUE_PARAMETER]))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)

    # Test with value being number
    params[CleanMissing.VALUE_PARAMETER] = 1200
    instance = CleanMissing(params, inputs, outputs,
                            named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = expected_code.replace('200', '1200')
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_clean_missing_missing_attribute_param_failure():
    inputs = ['input_1']
    outputs = ['output_1']
    with pytest.raises(ValueError):
        CleanMissing({}, inputs, outputs, named_inputs={}, named_outputs={})


def test_difference_minimal_params_success():
    params = {}
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    class_name = Difference
    instance = class_name(params, inputs, outputs,
                          named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = "output_1 = input_1.subtract(input_2)"
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_distinct_minimal_params_success():
    params = {}
    inputs = ['input_1']
    outputs = ['output_1']
    instance = Distinct(params, inputs, outputs,
                        named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = "output_1 = input_1.dropDuplicates()"
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_distinct_by_attributes_success():
    params = {
        'attributes': ['name']
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = Distinct(params, inputs, outputs,
                        named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = "output_1 = input_1.dropDuplicates(subset=['name'])"
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_drop_minimal_params_success():
    params = {
        'column': 'TEST'
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = Drop(params, inputs, outputs,
                    named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = "output_1 = input_1.drop('{}')".format(params['column'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_filter_minimum_params_success():
    params = {
        Filter.FILTER_PARAM: [{
            'attribute': 'code',
            'f': '>',
            'value': '201'
        }]
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = Filter(params, inputs, outputs,
                      named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = ("output_1 = input_1.filter("
                     "functions.col('{attribute}') {f} '{value}')").format(
        **params[Filter.FILTER_PARAM][0])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_filter_missing_parameter_filter_failure():
    params = {
    }
    inputs = ['input_1']
    outputs = ['output_1']
    with pytest.raises(ValueError):
        Filter(params, inputs, outputs,
               named_inputs={}, named_outputs={})


def test_intersection_minimal_params_success():
    params = {}
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    instance = Intersection(params, inputs, outputs,
                            named_inputs={}, named_outputs={})

    code = instance.generate_code()
    expected_code = "output_1 = input_1.intersect(input_2)"
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_join_inner_join_minimal_params_success():
    params = {
        'left_attributes': ['id', 'cod'],
        'right_attributes': ['id', 'cod']
    }
    named_inputs = {'input data 1': 'input_1', 'input data 2': 'input_2'}
    outputs = ['output_1']
    instance = Join(params, named_inputs.values(), outputs,
                    named_inputs=named_inputs, named_outputs={})

    code = instance.generate_code()
    expected_code = dedent("""
        cond_{out} = [{left_in}['id'] == {right_in}['id'], {left_in}['cod'] == {right_in}['cod']]
        {out} = {left_in}.join({right_in}, on=cond_{out}, how='{how}').drop({right_in}['id']).drop({right_in}['cod'])""".format(
        out=outputs[0],
        left_in=named_inputs['input data 1'],
        right_in=named_inputs['input data 2'], how="inner"))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_join_left_join_minimal_params_success():
    params = {
        'left_attributes': ['id', 'cod'],
        'right_attributes': ['id', 'cod'],
        Join.JOIN_TYPE_PARAM: 'left'
    }
    named_inputs = {'input data 1': 'input_1', 'input data 2': 'input_2'}
    outputs = ['output_1']
    instance = Join(params, named_inputs.values(), outputs,
                    named_inputs=named_inputs, named_outputs={})

    code = instance.generate_code()
    expected_code = dedent("""
        cond_{0} = [{1}['id'] == {2}['id'], {1}['cod'] == {2}['cod']]
        {0} = {1}.join({2}, on=cond_{0}, how='{3}').drop({2}['id'])\
            .drop({2}['cod'])""".format(
        outputs[0], named_inputs['input data 1'],
        named_inputs['input data 2'], params[Join.JOIN_TYPE_PARAM]))


    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_join_remove_right_columns_success():
    params = {
        'left_attributes': ['id', 'cod'],
        'right_attributes': ['id2', 'cod2'],
        Join.KEEP_RIGHT_KEYS_PARAM: 'False'
    }
    named_inputs = {'input data 1': 'input_1', 'input data 2': 'input_2'}
    outputs = ['output_1']
    instance = Join(params, named_inputs.values(), outputs,
                    named_inputs=named_inputs, named_outputs={})

    code = instance.generate_code()
    expected_code = dedent("""
        cond_{0} = [{1}['id'] == {2}['id2'], {1}['cod'] == {2}['cod2']]
        {0} = {1}.join({2}, on=cond_{0}, how='{3}').drop({2}['id2'])\
            .drop({2}['cod2'])
        """.format(outputs[0], named_inputs['input data 1'],
                   named_inputs['input data 2'], 'inner'))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_join_missing_left_or_right_param_failure():
    params = {
        'right_attributes': ['id', 'cod']
    }
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    with pytest.raises(ValueError):
        Join(params, inputs, outputs,
             named_inputs={}, named_outputs={})

    params = {
        'left_attributes': ['id', 'cod']
    }
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    with pytest.raises(ValueError):
        Join(params, inputs, outputs,
             named_inputs={}, named_outputs={})


def test_random_split_minimal_params_success():
    params = {
        'weights': '40',
        'seed': '1234321'
    }
    instance = RandomSplit(params, inputs=['input_1'],
                           outputs=['output_1', 'output_2'],
                           named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = """{0}, {1} = {2}.randomSplit({3}, {4})""".format(
        'output_1', 'output_2', 'input_1', '[40.0, 60.0]', 1234321)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sample_or_partition_minimal_params_success():
    params = {
        'withReplacement': 'False',
        'fraction': '0.3',
        'seed': '0'
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = SampleOrPartition(params, inputs, outputs,
                                 named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = "output_1 = input_1.sample(withReplacement={}, " \
                    "fraction={}, seed={})".format(params['withReplacement'],
                                                   params['fraction'],
                                                   params['seed'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sample_or_partition_type_value_success():
    params = {
        'withReplacement': 'False',
        'value': '400',
        'seed': '0',
        'type': SampleOrPartition.TYPE_VALUE
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = SampleOrPartition(params, inputs, outputs,
                                 named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = """output_1 = input_1.sample(withReplacement=False,
        fraction=1.0, seed=0).limit({})""".format(params['value'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sample_or_partition_type_head_success():
    params = {
        'withReplacement': 'False',
        'value': '365',
        'seed': '0',
        'type': SampleOrPartition.TYPE_HEAD
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = SampleOrPartition(params, inputs, outputs,
                                 named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = """output_1 = input_1.limit({})""".format(params['value'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    # print "\n\n", code, '\n\n', expected_code, '\n\n'
    assert result, msg


def test_sample_or_partition_invalid_fraction_failure():
    params = {
        'withReplacement': 'False',
        'fraction': '101',
        'seed': '0'
    }
    inputs = ['input_1']
    outputs = ['output_1']

    with pytest.raises(ValueError):
        SampleOrPartition(params, inputs, outputs, named_inputs={},
                          named_outputs={})


def test_sample_or_partition_fraction_percentage_success():
    params = {
        'withReplacement': 'False',
        'fraction': 45,
        'seed': '0'
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = SampleOrPartition(params, inputs, outputs,
                                 named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = "output_1 = input_1.sample(withReplacement={}, " \
                    "fraction={}, seed={})".format(params['withReplacement'],
                                                   params['fraction'] * 0.01,
                                                   params['seed'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sample_or_partition_fraction_missing_failure():
    params = {
        'withReplacement': 'False',
        'seed': '0'
    }
    inputs = ['input_1']
    outputs = ['output_1']
    with pytest.raises(ValueError):
        SampleOrPartition(params, inputs, outputs,
                          named_inputs={}, named_outputs={})


def test_select_minimal_params_success():
    params = {
        Select.ATTRIBUTES_PARAM: ['name', 'class']
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = Select(params, inputs, outputs, named_inputs={},
                      named_outputs={})

    code = instance.generate_code()
    expected_code = 'output_1 = input_1.select("name", "class")'

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_select_missing_attribute_param_failure():
    params = {
    }
    inputs = ['input_1']
    outputs = ['output_1']
    with pytest.raises(ValueError):
        Select(params, inputs, outputs, named_inputs={},
               named_outputs={})


def test_sort_minimal_params_success():
    params = {
        'attributes': [{'attribute': 'name', 'f': 'asc'},
                       {'attribute': 'class', 'f': 'desc'}],
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = Sort(params, inputs, outputs, named_inputs={},
                    named_outputs={})

    code = instance.generate_code()
    expected_code = 'output_1 = input_1.orderBy(["name", "class"], ' \
                    'ascending=[1, 0])'
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sort_missing_attributes_failure():
    params = {
    }
    inputs = ['input_1']
    outputs = ['output_1']
    with pytest.raises(ValueError) as excinfo:
        instance = Sort(params, inputs, outputs, named_inputs={},
                        named_outputs={})


def test_transformation_minumum_params_success():
    expr = {'tree': {
        "type": "CallExpression",
        "arguments": [
            {
                "type": "Literal",
                "value": "attr_name",
                "raw": "'attr_name'"
            }
        ],
        "callee": {
            "type": "Identifier",
            "name": "lower"
        }
    }, 'expression': "lower(attr_name)"}
    params = {
        Transformation.EXPRESSION_PARAM: json.dumps(expr),
        Transformation.ALIAS_PARAM: 'new_column',
        'input': 'input_x',
    }
    inputs = ['input_x']
    outputs = ['output_1']
    instance = Transformation(params, inputs, outputs,
                              named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = "output_1 = {input}.withColumn('{alias}'" \
                    ", functions.lower('attr_name'))".format(**params)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_transformation_math_expression_success():
    expr = {'tree': {
        "type": "BinaryExpression",
        "operator": "*",
        "left": {
            "type": "Identifier",
            "name": "a"
        },
        "right": {
            "type": "Literal",
            "value": 100,
            "raw": "100"
        }
    }, 'expression': "lower(a)"}

    params = {
        Transformation.EXPRESSION_PARAM: json.dumps(expr),
        Transformation.ALIAS_PARAM: 'new_column'
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = Transformation(params, inputs, outputs,
                              named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = "output_1 = input_1.withColumn('{alias}'" \
                    ", input_1.a * 100)".format(**params)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def xtest_transformation_complex_expression_success():
    expr = {'tree': {
        "type": "BinaryExpression",
        "operator": "+",
        "left": {
            "type": "UnaryExpression",
            "operator": "-",
            "argument": {
                "type": "Identifier",
                "name": "a"
            },
            "prefix": True
        },
        "right": {
            "type": "Identifier",
            "name": "b"
        }
    }, 'expression': "lower(a)"}

    params = {
        Transformation.EXPRESSION_PARAM: json.dumps(expr),
        Transformation.ALIAS_PARAM: 'new_column'
    }
    inputs = ['input_1']
    outputs = ['output_1']
    instance = Transformation(params, inputs, outputs,
                              named_inputs={}, named_outputs={})
    code = instance.generate_code()
    expected_code = "output_1 = input_1.withColumn('{alias}'" \
                    ", lower(col('a') + col('b')))".format(**params)

    debug_ast(code, expected_code)
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_transformation_missing_expr_failure():
    params = {
        Transformation.ALIAS_PARAM: 'new_column2'
    }
    inputs = ['input_1']
    outputs = ['output_1']
    with pytest.raises(ValueError):
        Transformation(params, inputs, outputs,
                       named_inputs={}, named_outputs={})


def test_transformation_missing_alias_failure():
    params = {
        Transformation.EXPRESSION_PARAM: '{}'
    }
    inputs = ['input_1']
    outputs = ['output_1']
    with pytest.raises(ValueError):
        Transformation(params, inputs, outputs,
                       named_inputs={}, named_outputs={})
