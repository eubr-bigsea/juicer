# -*- coding: utf-8 -*-
import ast
import json
from itertools import izip_longest
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.spark.ml_operation import FeatureIndexer, FeatureAssembler, \
                                      ApplyModel, EvaluateModel

from tests import compare_ast, format_code_comparison


def debug_ast(code, expected_code):
    print
    print code
    print '*' * 20
    print expected_code
    print '*' * 20

'''
 FeatureIndexer tests
'''


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
        FeatureIndexer.MAX_CATEGORIES_PARAM: -1,
    }
    inputs = ['input_1']
    outputs = ['output_1']

    with pytest.raises(ValueError):
        FeatureIndexer(params, inputs,
                       outputs, named_inputs={},
                       named_outputs={})


def test_feature_indexer_operation_failure():
    params = {}
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    with pytest.raises(ValueError):
        FeatureIndexer(params, inputs,
                       outputs, named_inputs={},
                       named_outputs={})

'''
 FeatureAssembler tests
'''


def test_feature_assembler_operation_success():
    params = {
        FeatureAssembler.ATTRIBUTES_PARAM: ['col'],
        FeatureAssembler.ALIAS_PARAM: 'c'
    }

    inputs = ['input_1']
    outputs = ['output_1']

    instance = FeatureAssembler(params, inputs,
                                outputs, named_inputs={},
                                named_outputs={})

    code = instance.generate_code()

    expected_code = dedent("""
            assembler = feature.VectorAssembler(inputCols={features},
                                                outputCol="{alias}")
            {input_1}_without_null = {input_1}.na.drop(subset={features})
            {output_1} = assembler.transform({input_1}_without_null)


            """.format(features=json.dumps(
                                    params[FeatureIndexer.ATTRIBUTES_PARAM]),
                       alias=params[FeatureAssembler.ALIAS_PARAM],
                       output_1=outputs[0],
                       input_1=inputs[0]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_feature_assembler_operation_failure():
    params = {
        # FeatureAssembler.ATTRIBUTES_PARAM: ['col'],
        FeatureAssembler.ALIAS_PARAM: 'c'
    }

    inputs = ['input_1']
    outputs = ['output_1']

    with pytest.raises(ValueError):
        FeatureAssembler(params, inputs,
                         outputs, named_inputs={},
                         named_outputs={})

'''
 ApplyModel tests
'''

def test_apply_model_operation_success():

    params = {}
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']

    instance = ApplyModel(params, inputs,
                          outputs, named_inputs={},
                          named_outputs={})

    code = instance.generate_code()

    expected_code = dedent("""
        {output_1} = {input_2}.transform({input_1})
        """.format(output_1=outputs[0],
                   input_1=inputs[0],
                   input_2=inputs[1]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_apply_model_operation_failure():

    params = {}
    inputs = ['input_1']
    # inputs = ['input_1', 'input_2']
    outputs = ['output_1']

    with pytest.raises(ValueError):
        apply_model = ApplyModel(params, inputs,
                                 outputs, named_inputs={},
                                 named_outputs={})
        apply_model.generate_code()


'''
    EvaluateModel tests
'''


def test_evaluate_model_operation_success():
    params = {

        EvaluateModel.PREDICTION_ATTRIBUTE_PARAM: 'c',
        EvaluateModel.LABEL_ATTRIBUTE_PARAM: 'c',
        EvaluateModel.METRIC_PARAM: 'f1',
    }
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']

    instance = EvaluateModel(params, inputs,
                             outputs, named_inputs={},
                             named_outputs={})

    code = instance.generate_code()

    expected_code = dedent("""
            # Creates the evaluator according to the model
            # (user should not change it)
            evaluator = {evaluator}({predic_col}='{predic_atr}',
                                  labelCol='{label_atr}', metricName='{metric}')

            {output} = evaluator.evaluate({input_1})
            """.format(output=outputs[0],
                       input_2=inputs[1],
                       input_1=inputs[0],
                       predic_atr=params[EvaluateModel.PREDICTION_ATTRIBUTE_PARAM],
                       label_atr=params[EvaluateModel.LABEL_ATTRIBUTE_PARAM],
                       metric=params[EvaluateModel.METRIC_PARAM],
                       evaluator=EvaluateModel.METRIC_TO_EVALUATOR[
                                            params[EvaluateModel.METRIC_PARAM]][0],
                       predic_col=EvaluateModel.METRIC_TO_EVALUATOR[
                                            params[EvaluateModel.METRIC_PARAM]][1]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


# @!WORKING HERE
def test_evaluate_model_operation_missing_output_param_failure():
    params = {

        EvaluateModel.PREDICTION_ATTRIBUTE_PARAM: 'c',
        EvaluateModel.LABEL_ATTRIBUTE_PARAM: 'c',
        EvaluateModel.METRIC_PARAM: 'f1',
    }
    inputs = ['input_1', 'input_2']
    outputs = []
    #
    # instance = EvaluateModel(params, inputs,
    #                          outputs, named_inputs={},
    #                          named_outputs={})
    with pytest.raises(ValueError):
        EvaluateModel(params, inputs,
                      outputs, named_inputs={},
                      named_outputs={})


def test_evaluate_model_operation_wrong_metric_param_failure():
    params = {

        EvaluateModel.PREDICTION_ATTRIBUTE_PARAM: 'c',
        EvaluateModel.LABEL_ATTRIBUTE_PARAM: 'c',
        EvaluateModel.METRIC_PARAM: 'mist',
    }
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    #
    # instance = EvaluateModel(params, inputs,
    #                          outputs, named_inputs={},
    #                          named_outputs={})
    with pytest.raises(ValueError):
        EvaluateModel(params, inputs,
                      outputs, named_inputs={},
                      named_outputs={})


def test_evaluate_model_operation_missing_metric_param_failure():
    params = {

        EvaluateModel.PREDICTION_ATTRIBUTE_PARAM: 'c',
        EvaluateModel.LABEL_ATTRIBUTE_PARAM: 'c',
        EvaluateModel.METRIC_PARAM: '',
    }
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    #
    # instance = EvaluateModel(params, inputs,
    #                          outputs, named_inputs={},
    #                          named_outputs={})
    with pytest.raises(ValueError):
        EvaluateModel(params, inputs,
                      outputs, named_inputs={},
                      named_outputs={})


def test_evaluate_model_operation_missing_prediction_attribute_failure():
    params = {

        EvaluateModel.PREDICTION_ATTRIBUTE_PARAM: '',
        EvaluateModel.LABEL_ATTRIBUTE_PARAM: 'c',
        EvaluateModel.METRIC_PARAM: 'f1',
    }
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    #
    # instance = EvaluateModel(params, inputs,
    #                          outputs, named_inputs={},
    #                          named_outputs={})
    with pytest.raises(ValueError):
        EvaluateModel(params, inputs,
                      outputs, named_inputs={},
                      named_outputs={})


def test_evaluate_model_operation_missing_label_attribute_failure():
    params = {

        EvaluateModel.PREDICTION_ATTRIBUTE_PARAM: 'c',
        EvaluateModel.LABEL_ATTRIBUTE_PARAM: '',
        EvaluateModel.METRIC_PARAM: 'f1',
    }
    inputs = ['input_1', 'input_2']
    outputs = ['output_1']
    #
    # instance = EvaluateModel(params, inputs,
    #                          outputs, named_inputs={},
    #                          named_outputs={})
    with pytest.raises(ValueError):
        EvaluateModel(params, inputs,
                      outputs, named_inputs={},
                      named_outputs={})