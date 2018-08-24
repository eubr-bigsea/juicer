# -*- coding: utf-8 -*-
import ast
from textwrap import dedent

import pytest
from juicer.sklearn.etl_operation import AddColumnsOperation, \
    SplitOperation, SortOperation, \
    DifferenceOperation, DistinctOperation, IntersectionOperation, \
    JoinOperation, DropOperation, \
    TransformationOperation, SelectOperation, AggregationOperation, \
    FilterOperation, \
    CleanMissingOperation, \
    UnionOperation, \
    SampleOrPartitionOperation, ReplaceValuesOperation


from tests import compare_ast, format_code_comparison


def debug_ast(code, expected_code):
    print("""
    Code
    {sep}
    {code}
    {sep}
    Expected
    {sep}
    {expected}
    """.format(code=code, sep='-' * 20, expected=expected_code))


def test_add_columns_minimum_params_success():
    params = {}
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}

    instance = AddColumnsOperation(parameters=params,
                                   named_inputs=n_in,
                                   named_outputs=n_out)

    code = instance.generate_code()
    print (code)
    expected_code = dedent("""
    out = pd.merge(df1, df2, left_index=True, 
        right_index=True, suffixes=('ds0_', 'ds0_'))
    """.format(
        out=n_out['output data'],
        in0=n_in['input data 1'],
        in1=n_in['input data 2']))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


#############################################################################
#   Difference Operation
def test_difference_minimal_params_success():
    params = {}
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}
    class_name = DifferenceOperation
    instance = class_name(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    expected_code = "cols = {in1}.columns\n" \
                    "{out} = pd.merge({in1}, {in2}, " \
                    "indicator=True, how='left', on=None)\n" \
                    "{out} = {out}.loc[{out}['_merge'] == 'left_only', cols]"\
        .format(out=n_out['output data'], in1=n_in['input data 1'],
                in2=n_in['input data 2'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


#############################################################################
#   Distinct Operation
def test_remove_duplicated_minimal_params_success():
    params = {}
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = DistinctOperation(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "{out} = {in1}.drop_duplicates(subset=None, keep='first')"\
        .format(out=n_out['output data'], in1=n_in['input data'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_remove_duplicated_by_attributes_success():
    params = {
        'attributes': ['name']
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = DistinctOperation(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "{} = {}.drop_duplicates(subset={}, keep='first')"\
        .format(n_out['output data'], n_in['input data'], params['attributes'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


#############################################################################
#   Drop Operation
def test_drop_minimal_params_success():
    params = {
        'attributes': 'TEST'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = DropOperation(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "{out} = {input}.drop(columns={columns})" \
        .format(out=n_out['output data'], input=n_in['input data'],
                columns=params['attributes'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


#############################################################################
#   Intersection Operation
def test_intersection_minimal_params_success():
    params = {}
    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}
    instance = IntersectionOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent(
        """
        if len({in1}.columns) != len({in2}.columns):
            raise ValueError('{error}')
        {in1} = {in1}.dropna(axis=0, how='any')
        {in2} = {in2}.dropna(axis=0, how='any')
        keys = {in1}.columns.tolist()
        {in1} = pd.merge({in1}, {in2}, how='left', on=keys, 
        indicator=True, copy=False)
        {out} = {in1}.loc[{in1}['_merge'] == 'both', keys]
        """.format(
            out=n_out['output data'], in1=n_in['input data 1'],
            in2=n_in['input data 2'],
            error=(
                'For intersection operation, both input data '
                'sources must have the same number of attributes '
                'and types.')))
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


#############################################################################
# Replace Values Operation
def test_replace_value_minimal_params_success():
    params = {
        "attributes": ["col1", "col2"],
        "replacement": 10,
        "value": -10
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    instance = ReplaceValuesOperation(params, named_inputs=n_in,
                                      named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent(
        """
        output_1 = input_1
        replacement = {replaces}
        for col in replacement:
            list_replaces = replacement[col]
            output_1[col] = output_1[col].replace(list_replaces[0],
            list_replaces[1])
        """.format(out=n_out['output data'], in1=n_in['input data'],
                   replaces={"col2": [[-10], [10]], "col1": [[-10], [10]]}))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_replace_value_missing_attribute_param_failure():
    params = {
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        ReplaceValuesOperation(params, named_inputs=n_in, named_outputs=n_out)


#############################################################################
#   Sample Operation
def test_sample_or_partition_minimal_params_success():
    params = {
        'fraction': '3',
        'seed': '0'
    }
    n_in = {'input data': 'input_1'}
    n_out = {'sampled data': 'output_1'}
    instance = SampleOrPartitionOperation(params, named_inputs=n_in,
                                          named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "output_1 = input_1.sample(frac={}, random_state={})"\
        .format('0.03', params['seed'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sample_or_partition_type_value_success():
    params = {
        'value': '400',
        'seed': '0',
        'type': SampleOrPartitionOperation.TYPE_VALUE
    }
    n_in = {'input data': 'input_1'}
    n_out = {'sampled data': 'output_1'}
    instance = SampleOrPartitionOperation(params, named_inputs=n_in,
                                          named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "output_1 = input_1.sample(n={}, random_state=0)"\
        .format(params['value'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sample_or_partition_type_head_success():
    params = {
        'value': 365,
        'seed': 0,
        'type': SampleOrPartitionOperation.TYPE_HEAD
    }
    n_in = {'input data': 'input_1'}
    n_out = {'sampled data': 'output_1'}
    instance = SampleOrPartitionOperation(params, named_inputs=n_in,
                                          named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "output_1 = input_1.head({})".format(params['value'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg


def test_sample_or_partition_invalid_value_failure():
    params = {
        'value': -365,
        'seed': '0',
        'type': SampleOrPartitionOperation.TYPE_HEAD
    }

    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        SampleOrPartitionOperation(params, named_inputs=n_in,
                                   named_outputs=n_out)


def test_sample_or_partition_invalid_fraction_failure():
    params = {
        'fraction': '101',
        'seed': '0'
    }

    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        SampleOrPartitionOperation(params, named_inputs=n_in,
                                   named_outputs=n_out)


def test_sample_or_partition_fraction_percentage_success():
    params = {
        'fraction': 45
    }
    n_in = {'input data': 'input_1'}
    n_out = {'sampled data': 'output_1'}
    instance = SampleOrPartitionOperation(params, named_inputs=n_in,
                                          named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "output_1 = input_1.sample(frac={}, random_state={})"\
        .format(params['fraction'] * 0.01, 'None')
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_sample_or_partition_fraction_missing_failure():
    params = {
        'seed': '0'
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        SampleOrPartitionOperation(params, named_inputs=n_in,
                                   named_outputs=n_out)


#############################################################################
# Select Operation
def test_select_minimal_params_success():
    params = {
        SelectOperation.ATTRIBUTES_PARAM: ['name', 'class']
    }
    n_in = {'input data': 'input_1'}
    n_out = {'output projected data': 'output_1'}
    instance = SelectOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    columns = ', '.join(
        ['"{}"'.format(x) for x in params[SelectOperation.ATTRIBUTES_PARAM]])
    expected_code = '{out} = {in1}[{columns}]'\
        .format(out=n_out['output projected data'],
                in1=n_in['input data'], columns=columns)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


def test_select_missing_attribute_param_failure():
    params = {
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        SelectOperation(params, named_inputs=n_in, named_outputs=n_out)


#############################################################################
# Sort Operation
def test_sort_minimal_params_success():
    params = {
        'attributes': [{'attribute': 'name', 'f': 'asc'},
                       {'attribute': 'class', 'f': 'desc'}],
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = SortOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()
    code1 = "{out} = {in0}.sort_values(by=['name', 'class'], " \
            "ascending=[True, False])"\
        .format(out=n_out['output data'], in0=n_in['input data'])
    result, msg = compare_ast(ast.parse(code), ast.parse(code1))
    assert result, msg


@pytest.mark.xfail(raises=UnboundLocalError)
def test_sort_missing_attributes_failure():
    params = {}
    with pytest.raises(ValueError):
        n_in = {'input data': 'df1'}
        n_out = {'output data': 'out'}
        SortOperation(params, named_inputs=n_in, named_outputs=n_out)


#############################################################################
# Split Operation
def test_random_split_params_success():
    params = {
        'weights': '40.0',
        'seed': '1234321'
    }
    n_in = {'input data': 'df1'}
    n_out = {'splitted data 1': 'out1', 'splitted data 2': 'out2'}

    instance = SplitOperation(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()

    expected_code = """{out1}, {out2} = np.split({input}.sample(frac=1, 
    random_state={seed}), [int({weights}*len({input}))])
    """.format(out1=n_out['splitted data 1'], out2=n_out['splitted data 2'],
               input=n_in['input data'], weights='0.4', seed=1234321)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


#############################################################################
# Transformation Operation
def test_transformation_minumum_params_success():

    params = {
        "expression": [
            {"alias": "new_col1", "expression": "col1+2*9",
             "tree": {"operator": "+", "right": {"operator": "*",
                                                 "right": {"raw": "9",
                                                           "type": "Literal",
                                                           "value": 9},
                                                 "type": "BinaryExpression",
                                                 "left": {"raw": "2",
                                                          "type": "Literal",
                                                          "value": 2}},
                      "type": "BinaryExpression",
                      "left": {"type": "Identifier", "name": "col1"}},
             "error": 'null'},
            {"alias": "new_col2", "expression": "len(col2, 3)",
                              "tree": {"type": "CallExpression",
                                       "callee": {"type": "Identifier",
                                                  "name": "len"}, "arguments": [
                                      {"type": "Identifier", "name": "col2"},
                                      {"raw": "3", "type": "Literal",
                                       "value": 3}]}, "error": 'null'},
            {"alias": "new_col3", "expression": "split(col3, ',')",
             "tree": {"type": "CallExpression",
                      "callee": {"type": "Identifier", "name": "split"},
                      "arguments": [{"type": "Identifier", "name": "col3"},
                                    {"raw": "','", "type": "Literal",
                                     "value": ","}]}, "error": 'null'}
        ]
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = TransformationOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)
    code = instance.generate_code()

    expected_code = dedent(
        """
        functions = [{expr}]
        {out} = {in1}
        for col, function, imp in functions:
            exec(imp)
            function = eval(function)
            {out}[col] = {out}[col].apply(function, axis=1)
        """.format(out=n_out['output data'],
                   in1=n_in['input data'], alias='result_1',
                   expr=[
                       ['new_col1', "lambda row: row['col1'] + 2 * 9", ''],
                       ['new_col2', "lambda row: len(row['col2'], 3)", ''],
                       ['new_col3', "lambda row: row['col3'].split(',')", '']
                   ])
    )

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_transformation_math_expression_success():
    alias = 'result_2'
    expr = [{'tree': {
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
    }, 'alias': alias, 'expression': "lower(a)"}]

    params = {
        TransformationOperation.EXPRESSION_PARAM: expr,
    }
    n_in = {'input data': 'df1'}
    n_out = {'output data': 'out'}
    instance = TransformationOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)
    code = instance.generate_code()

    expected_code = dedent(
        """
        functions = [{expr}]
        {out} = {in1}
        for col, function, imp in functions:
            exec(imp)
            function = eval(function)
            {out}[col] = {out}[col].apply(function, axis=1)
        """.format(out=n_out['output data'],
                   in1=n_in['input data'], alias='result_2',
                   expr=[
                       ['result_2', "lambda row: row['a'] * 100", '']
                   ])
        )

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_transformation_missing_expr_failure():
    params = {
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'df1'}
        n_out = {'output data': 'out'}
        TransformationOperation(params, named_inputs=n_in,
                                named_outputs=n_out)


#############################################################################
# Union (Add-Rows) Operation
def test_union_minimal_params_success():
    params = {}

    n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
    n_out = {'output data': 'out'}

    instance = UnionOperation(params, named_inputs=n_in, named_outputs=n_out)
    code = instance.generate_code()
    expected_code = "{out} = pd.concat([{in0}, {in1}], " \
                    "sort=False, axis=0, ignore_index=True)"\
        .format(out=n_out['output data'], in0=n_in['input data 1'],
                in1=n_in['input data 2'])
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg


# def test_aggregation_rows_minimal_params_success():
#     params = {
#         AggregationOperation.FUNCTION_PARAM: [
#             {'attribute': 'income', 'f': 'AVG', 'alias': 'avg_income'}],
#         AggregationOperation.ATTRIBUTES_PARAM: ['country']
#     }
#     n_in = {'input data': 'input_1'}
#     n_out = {'output data': 'output_1'}
#
#     instance = AggregationOperation(params, named_inputs=n_in,
#                                     named_outputs=n_out)
#     code = instance.generate_code()
#
#     expected_code = dedent("""
#          pivot_values = None
#          pivot_attr = ''
#          if pivot_attr:
#               {out} = {in0}.groupBy(
#                  functions.col('{agg}')).pivot(
#                      pivot_attr, pivot_values).agg(
#                          functions.avg('income').alias('avg_income'))
#          else:
#               {out} = {in0}.groupBy(
#                  functions.col('{agg}')).agg(
#                      functions.avg('income').alias('avg_income'))
#
#         """.format(out=n_out['output data'], in0=n_in['input data'],
#                    agg='country', ))
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_aggregation_rows_group_all_missing_attributes_success():
#     params = {
#         AggregationOperation.FUNCTION_PARAM: [
#             {'attribute': 'income', 'f': 'AVG', 'alias': 'avg_income'}],
#     }
#     n_in = {'input data': 'input_1'}
#     n_out = {'output data': 'output_1'}
#
#     instance = AggregationOperation(params, named_inputs=n_in,
#                                     named_outputs=n_out)
#     code = instance.generate_code()
#
#     expected_code = """{out} = {in0}.agg(
#                         functions.avg('income').alias('avg_income'))""".format(
#         out=n_out['output data'], in0=n_in['input data'], agg='country', )
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#     assert result, msg
#
#
# def test_aggregation_missing_function_param_failure():
#     params = {
#         AggregationOperation.ATTRIBUTES_PARAM: ['country']
#     }
#     n_in = {'input data': 'input_1'}
#     n_out = {'output data': 'output_1'}
#     with pytest.raises(ValueError):
#         AggregationOperation(params, named_inputs=n_in,
#                              named_outputs=n_out)
#
#
# def test_clean_missing_minimal_params_success():
#     params = {
#         CleanMissingOperation.ATTRIBUTES_PARAM: ['name'],
#         CleanMissingOperation.MIN_MISSING_RATIO_PARAM: "0.0",
#         CleanMissingOperation.MAX_MISSING_RATIO_PARAM: "1.0",
#     }
#     n_in = {'input data': 'input_1'}
#     n_out = {'output result': 'output_1'}
#     instance = CleanMissingOperation(params, named_inputs=n_in,
#                                      named_outputs=n_out)
#     code = instance.generate_code()
#     expected_code = dedent("""
#     ratio_{input_1} = {input_1}.select(
#         (functions.avg(functions.col('{attribute}').isNull().cast(
#         'int'))).alias('{attribute}')).collect()
#     attributes_{input_1} = [c for c in ["{attribute}"]
#                  if 0.0 <= ratio_{input_1}[0][c] <= 1.0]
#     if len(attributes_input_1) > 0:
#         {output_1} = {input_1}.na.drop(how='any', subset=attributes_{input_1})
#     else:
#         {output_1} = {input_1}
#     """.format(input_1=n_in['input data'], attribute=params['attributes'][0],
#                output_1=n_out['output result']))
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_clean_missing_without_missing_rating_params_success():
#     params = {
#         CleanMissingOperation.ATTRIBUTES_PARAM: ['name'],
#     }
#     n_in = {'input data': 'input_1'}
#     n_out = {'output result': 'output_1'}
#     instance = CleanMissingOperation(params, named_inputs=n_in,
#                                      named_outputs=n_out)
#     code = instance.generate_code()
#     expected_code = dedent("""
#     attributes_{input_1} = ['{attribute}']
#     if len(attributes_input_1) > 0:
#         {output_1} = {input_1}.na.drop(how='any', subset=attributes_{input_1})
#     else:
#         {output_1} = {input_1}
#     """.format(input_1=n_in['input data'], attribute=params['attributes'][0],
#                output_1=n_out['output result']))
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_clean_missing_minimal_params_type_value_success():
#     params = {
#         CleanMissingOperation.ATTRIBUTES_PARAM: ['name'],
#         CleanMissingOperation.MIN_MISSING_RATIO_PARAM: "0.0",
#         CleanMissingOperation.MAX_MISSING_RATIO_PARAM: "1.0",
#         CleanMissingOperation.VALUE_PARAMETER: "200",
#         CleanMissingOperation.CLEANING_MODE_PARAM: CleanMissingOperation.VALUE
#     }
#     n_in = {'input data': 'input_1'}
#     n_out = {'output result': 'output_1'}
#     instance = CleanMissingOperation(params, named_inputs=n_in,
#                                      named_outputs=n_out)
#     code = instance.generate_code()
#     expected_code = dedent("""
#     ratio_{input_1} = {input_1}.select(
#         (functions.avg(functions.col('{attribute}').isNull().cast(
#         'int'))).alias('{attribute}')).collect()
#     attributes_{input_1} = [c for c in ["{attribute}"]
#                  if 0.0 <= ratio_{input_1}[0][c] <= 1.0]
#     if len(attributes_input_1) > 0:
#         {output_1} = {input_1}.na.fill(value={value},
#                 subset=attributes_{input_1})
#     else:
#         {output_1} = {input_1}
#     """.format(input_1=n_in['input data'], attribute=params['attributes'][0],
#                output_1=n_out['output result'],
#                value=params[CleanMissingOperation.VALUE_PARAMETER]))
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#     assert result, msg + format_code_comparison(code, expected_code)
#
#     # Test with value being number
#     params[CleanMissingOperation.VALUE_PARAMETER] = 1200
#     instance = CleanMissingOperation(params, named_inputs=n_in,
#                                      named_outputs=n_out)
#     code = instance.generate_code()
#     expected_code = expected_code.replace('200', '1200')
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_clean_missing_missing_attribute_param_failure():
#     params = {}
#     with pytest.raises(ValueError):
#         n_in = {'input data': 'input_1'}
#         n_out = {'output data': 'output_1'}
#         CleanMissingOperation(params, named_inputs=n_in,
#                               named_outputs=n_out)
#

# def test_filter_minimum_params_success():
#     params = {
#         FilterOperation.FILTER_PARAM: [{
#             'attribute': 'code',
#             'f': '>',
#             'value': '201'
#         }],
#         'config': {
#
#         }
#     }
#     n_in = {'input data': 'input_1'}
#     n_out = {'output data': 'output_1'}
#     instance = FilterOperation(params, named_inputs=n_in, named_outputs=n_out)
#
#     code = instance.generate_code()
#     expected_code = ("{out} = {in1}.filter("
#                      "functions.col('{attribute}') {f} '{value}')").format(
#         out=n_out['output data'], in1=n_in['input data'],
#         **params[FilterOperation.FILTER_PARAM][0])
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_filter_missing_parameter_filter_failure():
#     params = {
#     }
#     with pytest.raises(ValueError):
#         n_in = {'input data': 'input_1'}
#         n_out = {'output data': 'output_1'}
#         FilterOperation(params, named_inputs=n_in, named_outputs=n_out)
#
#

# def test_join_inner_join_minimal_params_success():
#     params = {
#         'left_attributes': ['id', 'cod'],
#         'right_attributes': ['id', 'cod'],
#         'aliases': 'left_, right_  '
#     }
#     n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
#     n_out = {'output data': 'out'}
#     instance = JoinOperation(params, named_inputs=n_in, named_outputs=n_out)
#
#     code = instance.generate_code()
#     expected_code = dedent("""
#         def _rename_attributes(df, prefix):
#             result = df
#             for col in df.columns:
#                 result = result.withColumnRenamed(col, '{{}}{{}}'.format(
#                     prefix, col))
#             return result
#
#         in0_renamed = _rename_attributes({in0}, '{a0}')
#         in1_renamed = _rename_attributes({in1}, '{a1}')
#
#         condition = [in0_renamed['{a0}id'] == in1_renamed['{a1}id'],
#             in0_renamed['{a0}cod'] == in1_renamed['{a1}cod']]
#
#         {out} = in0_renamed.join(in1_renamed, on=condition, how='{how}').drop(
#             in1_renamed['{a1}id']).drop(in1_renamed['{a1}cod'])""".format(
#         out=n_out['output data'], in0=n_in['input data 1'],
#         a0='left_', a1='right_',
#         in1=n_in['input data 2'], how="inner"))
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_join_left_join_keep_columns_minimal_params_success():
#     params = {
#         'left_attributes': ['id', 'cod'],
#         'right_attributes': ['id', 'cod'],
#         JoinOperation.JOIN_TYPE_PARAM: 'left',
#         JoinOperation.KEEP_RIGHT_KEYS_PARAM: True,
#         'aliases': 'left_, right_  '
#     }
#     n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
#     n_out = {'output data': 'out'}
#     instance = JoinOperation(params, named_inputs=n_in, named_outputs=n_out)
#
#     code = instance.generate_code()
#     expected_code = dedent("""
#         def _rename_attributes(df, prefix):
#             result = df
#             for col in df.columns:
#                 result = result.withColumnRenamed(col, '{{}}{{}}'.format(
#                     prefix, col))
#             return result
#
#         in0_renamed = _rename_attributes({in0}, '{a0}')
#         in1_renamed = _rename_attributes({in1}, '{a1}')
#
#         condition = [in0_renamed['{a0}id'] == in1_renamed['{a1}id'],
#             in0_renamed['{a0}cod'] == in1_renamed['{a1}cod']]
#         {out} = in0_renamed.join(in1_renamed, on=condition, how='left')
#         """.format(
#         out=n_out['output data'], in0=n_in['input data 1'],
#         a0='left_', a1='right_',
#         in1=n_in['input data 2'], type=params[JoinOperation.JOIN_TYPE_PARAM], ))
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_join_remove_right_columns_success():
#     params = {
#         'left_attributes': ['id', 'cod'],
#         'right_attributes': ['id2', 'cod2'],
#         JoinOperation.KEEP_RIGHT_KEYS_PARAM: 'False',
#         'aliases': 'left_, right_  '
#     }
#     n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
#     n_out = {'output data': 'out'}
#     instance = JoinOperation(params, named_inputs=n_in, named_outputs=n_out)
#
#     code = instance.generate_code()
#     expected_code = dedent("""
#         def _rename_attributes(df, prefix):
#             result = df
#             for col in df.columns:
#                 result = result.withColumnRenamed(col, '{{}}{{}}'.format(
#                     prefix, col))
#             return result
#         in0_renamed = _rename_attributes({in0}, '{a0}')
#         in1_renamed = _rename_attributes({in1}, '{a1}')
#
#         condition = [in0_renamed['{a0}id'] == in1_renamed['{a1}id2'],
#             in0_renamed['{a0}cod'] == in1_renamed['{a1}cod2']]
#         {out} = in0_renamed.join(in1_renamed, on=condition, how='inner')\\
#           .drop(in1_renamed['{a1}id2']).drop(in1_renamed['{a1}cod2'])""".format(
#         out=n_out['output data'], in0=n_in['input data 1'],
#         in1=n_in['input data 2'], a0='left_', a1='right_'))
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_join_missing_left_or_right_param_failure():
#     params = {
#         'right_attributes': ['id', 'cod']
#     }
#     with pytest.raises(ValueError):
#         n_in = {'input data 1': 'df1', 'input data 2': 'df2'}
#         n_out = {'output data': 'out'}
#         JoinOperation(params, named_inputs=n_in, named_outputs=n_out)
#
#     params = {
#         'left_attributes': ['id', 'cod']
#     }
#     with pytest.raises(ValueError):
#         JoinOperation(params, named_inputs=n_in, named_outputs=n_out)
