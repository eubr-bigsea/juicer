# -*- coding: utf-8 -*-
import ast
import json
from itertools import izip_longest
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.spark.ml_operation import FeatureIndexer

from tests import compare_ast, format_code_comparison


def debug_ast(code, expected_code):
    print
    print code
    print '*' * 20
    print expected_code
    print '*' * 20



# FeatureIndexer test
def test_feature_indexer_operation_success():
    params = {
        FeatureIndexer.TYPE_PARAM: 'string',
        FeatureIndexer.ATTRIBUTES_PARAM: ['col'],
        FeatureIndexer.ALIAS_PARAM: 'c',
        FeatureIndexer.MAX_CATEGORIES_PARAM: '20',
    }
    inputs = ['input_1']
    outputs = ['output_1']

    instance = FeatureIndexer(params, inputs,
                              outputs, named_inputs={},
                              named_outputs={})

    code = instance.generate_code()

    expected_code = dedent("""
        col_alias = dict({3})
        indexers = [feature.StringIndexer(inputCol=col, outputCol=alias,
                            handleInvalid='skip')
                    for col, alias in col_alias.iteritems()]

        # Use Pipeline to process all attributes once
        pipeline = Pipeline(stages=indexers)
        models = dict([(col[0], indexers[i].fit({1}))
                  for i, col in enumerate(col_alias)])

        # labels = [model.labels for model in models.itervalues()]
        # Spark ML 2.0.1 do not deal with null in indexer.
        # See SPARK-11569

        # input_1_without_null = input_1.na.fill('NA', subset=col_alias.keys())
        {1}_without_null = {1}.na.fill('NA', subset=col_alias.keys())

        # output_1 = pipeline.fit(input_1_without_null).transform(input_1_without_null)
        {2} = pipeline.fit({1}_without_null).transform({1}_without_null)

        """.format(params[FeatureIndexer.ATTRIBUTES_PARAM], inputs[0], outputs[0],
                   json.dumps(zip(params[FeatureIndexer.ATTRIBUTES_PARAM],
                                  params[FeatureIndexer.ALIAS_PARAM]))))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg


def test_feature_indexer_string_type_param_operation_failure():
    params = {
        FeatureIndexer.ATTRIBUTES_PARAM: ['col'],
        FeatureIndexer.TYPE_PARAM: 'XxX',
        FeatureIndexer.ALIAS_PARAM: 'c',
        FeatureIndexer.MAX_CATEGORIES_PARAM: '20',
    }
    inputs = ['input_1']
    outputs = ['output_1']

    instance = FeatureIndexer(params, inputs,
                             outputs, named_inputs={},
                             named_outputs={})
    # import pdb
    # pdb.set_trace()
    with pytest.raises(ValueError):
        indexer = FeatureIndexer(params, inputs,
                       outputs, named_inputs={},
                       named_outputs={})
        indexer.generate_code()


def test_feature_indexer_string_missing_attribute_param_operation_failure():
    params = {
        FeatureIndexer.TYPE_PARAM: 'string',
        FeatureIndexer.ALIAS_PARAM: 'c',
        FeatureIndexer.MAX_CATEGORIES_PARAM: '20',
    }
    inputs = ['input_1']
    outputs = ['output_1']

    # instance = FeatureIndexer(params, inputs,
    #                          outputs, named_inputs={},
    #                          named_outputs={})
    with pytest.raises(ValueError):
        FeatureIndexer(params, inputs,
                   outputs, named_inputs={},
                   named_outputs={})

def test_feature_indexer_vector_missing_attribute_param_operation_failure():
    params = {
        FeatureIndexer.TYPE_PARAM: 'string',
        FeatureIndexer.ALIAS_PARAM: 'c',
        FeatureIndexer.MAX_CATEGORIES_PARAM: '20',
    }
    inputs = ['input_1']
    outputs = ['output_1']

    # instance = FeatureIndexer(params, inputs,
    #                          outputs, named_inputs={},
    #                          named_outputs={})
    with pytest.raises(ValueError):
        FeatureIndexer(params, inputs,
                       outputs, named_inputs={},
                       named_outputs={})



def test_feature_indexer_vector_operation_success():
    params = {

        FeatureIndexer.TYPE_PARAM: 'vector',
        FeatureIndexer.ATTRIBUTES_PARAM: ['col'],
        FeatureIndexer.ALIAS_PARAM: 'c',
        FeatureIndexer.MAX_CATEGORIES_PARAM: 20,
    }
    inputs = ['input_1']
    outputs = ['output_1']

    instance = FeatureIndexer(params, inputs,
                              outputs, named_inputs={},
                              named_outputs={})

    code = instance.generate_code()

    expected_code = dedent("""
            col_alias = dict({3})
            indexers = [feature.VectorIndexer(maxCategories={4},
                            inputCol=col, outputCol=alias)
                            for col, alias in col_alias.iteritems()]

            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=indexers)
            models = dict([(col[0], indexers[i].fit({1})) for i, col in
                        enumerate(col_alias)])
            labels = None

            # Spark ML 2.0.1 do not deal with null in indexer.
            # See SPARK-11569
            {1}_without_null = {1}.na.fill('NA', subset=col_alias.keys())

            {2} = pipeline.fit({1}_without_null).transform({1}_without_null)
            """.format(params[FeatureIndexer.ATTRIBUTES_PARAM], inputs[0],
                       outputs[0],
                       json.dumps(zip(params[FeatureIndexer.ATTRIBUTES_PARAM],
                                      params[FeatureIndexer.ALIAS_PARAM])),
                       params[FeatureIndexer.MAX_CATEGORIES_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_feature_indexer_vector_operation_failure():
    params = {

        FeatureIndexer.TYPE_PARAM: 'vector',
        FeatureIndexer.ATTRIBUTES_PARAM: ['col'],
        FeatureIndexer.ALIAS_PARAM: 'c',
        FeatureIndexer.MAX_CATEGORIES_PARAM: None,
    }
    inputs = ['input_1']
    outputs = ['output_1']

    instance = FeatureIndexer(params, inputs,
                              outputs, named_inputs={},
                              named_outputs={})

    code = instance.generate_code()

    expected_code = dedent("""
            col_alias = {3}
            tokenizers = [Tokenizer(inputCol=col, outputCol=alias)
                                for col, alias in col_alias]

            # Use Pipeline to process all attributes once
            pipeline = Pipeline(stages=tokenizers)

            {2} = pipeline.fit({1}).transform({1})
        """.format(params[FeatureIndexer.ATTRIBUTES_PARAM], inputs[0], outputs[0],
                   json.dumps(zip(params[FeatureIndexer.ATTRIBUTES_PARAM],
                                  params[FeatureIndexer.ALIAS_PARAM]))))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_feature_indexer_operation_failure():
    params = {}
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    with pytest.raises(ValueError):
        FeatureIndexer(params, inputs,
                       outputs, named_inputs={},
                       named_outputs={})
