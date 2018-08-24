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

