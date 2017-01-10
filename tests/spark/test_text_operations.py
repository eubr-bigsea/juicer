# -*- coding: utf-8 -*-
import ast
import json
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.spark.text_operation import TokenizerOperation

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


# Test TokenizerOperation

def test_tokenizeroperation_type_simple():

    # TYPE_PARAM = 'type'
    # ATTRIBUTES_PARAM = 'attributes'
    # ALIAS_PARAM = 'alias'
    # EXPRESSION_PARAM = 'expression'
    # MINIMUM_SIZE = 'min_token_length'
    # TYPE_SIMPLE = 'simple'
    # TYPE_REGEX = 'regex'

    params = {
        'type' : 'simple',
        'attributes': 'col',
        'alias': 'col_alias'
    }
    inputs = ['input_1']
    outputs = ['output_1']

    instance = TokenizerOperation(params, inputs, outputs,
                                  named_inputs={}, named_outputs={})

    code = instance.generate_code()
    expected_code = """
            col_alias = {3}
            tokenizers = [Tokenizer(inputCol=col, outputCol=alias)
                                for col, alias in col_alias]

            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=tokenizers)

            {2} = pipeline.fit({1}).transform({1})
        """.format(params['attributes'],inputs[0], outputs[0],
                   json.dumps(zip(params['attributes'], params['alias'])))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg

def test_tokenizeroperation_type_regexp():
    params = {
        'type' : 'regex',
        'attributes': 'col',
        'alias': 'col_alias',
        'min_token_length': 3,
        'expression': r'\s+'
    }
    inputs = ['input_1']
    outputs = ['output_1']

    instance = TokenizerOperation(params, inputs, outputs,
                                  named_inputs={}, named_outputs={})

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    expected_code = """"""

    assert result, msg

# Test RemoveStopWordsOperation

# Test WordToVectorOperation

# Test NGramOperations