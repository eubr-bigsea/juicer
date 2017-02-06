# -*- coding: utf-8 -*-
import ast
import json
from itertools import izip_longest
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.spark.ml_operation import FeatureIndexer, FeatureAssembler, \
                                      ApplyModel, EvaluateModel, \
                                      CrossValidationOperation, ClassificationModel, \
                                      ClassifierOperation, SvmClassifierOperation

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


# @!BUG - Acessing 'task''order' in parameter attribute, but doesn't exist
# def test_evaluate_model_operation_missing_output_param_failure():
#     params = {
#
#         EvaluateModel.PREDICTION_ATTRIBUTE_PARAM: 'c',
#         EvaluateModel.LABEL_ATTRIBUTE_PARAM: 'c',
#         EvaluateModel.METRIC_PARAM: 'f1',
#     }
#     inputs = ['input_1', 'input_2']
#     outputs = []
#     with pytest.raises(ValueError):
#         evaluator = EvaluateModel(params, inputs,
#                                   outputs, named_inputs={},
#                                   named_outputs={})
#         evaluator.generate_code()


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


'''
    CrossValidationOperation Tests
'''

def test_cross_validation_partial_operation_success():
    params = {

        CrossValidationOperation.NUM_FOLDS_PARAM: 3,

    }
    inputs = ['df_1', 'df_2','df_3']
    named_inputs={'algorithm': 'df_1',
                  'input data': 'df_2',
                  'evaluator': 'df_3'}
    named_outputs = {'output_cv': 'output_1'}
    outputs = ['output_1']

    instance = CrossValidationOperation(params, inputs,
                                        outputs,
                                        named_inputs={'algorithm': 'df_1',
                                                      'input data': 'df_2',
                                                      'evaluator': 'df_3'},
                                        named_outputs={})

    code = instance.generate_code()

    expected_code = dedent("""
            grid_builder = tuning.ParamGridBuilder()
            estimator, param_grid = {algorithm}

            # if estimator.__class__ == classification.LinearRegression:
            #     param_grid = estimator.maxIter
            # elif estimator.__class__  == classification.:
            #     pass
            # elif estimator.__class__ == classification.DecisionTreeClassifier:
            #     # param_grid = (estimator.maxDepth, [2,3,4,5,6,7,8,9])
            #     param_grid = (estimator.impurity, ['gini', 'entropy'])
            # elif estimator.__class__ == classification.GBTClassifier:
            #     pass
            # elif estimator.__class__ == classification.RandomForestClassifier:
            #     param_grid = estimator.maxDepth
            for param_name, values in param_grid.iteritems():
                param = getattr(estimator, param_name)
                grid_builder.addGrid(param, values)

            evaluator = {evaluator}

            cross_validator = tuning.CrossValidator(
                estimator=estimator, estimatorParamMaps=grid_builder.build(),
                evaluator=evaluator, numFolds={folds})
            cv_model = cross_validator.fit({input_data})
            evaluated_data = cv_model.transform({input_data})
            best_model_{output}  = cv_model.bestModel
            metric_result = evaluator.evaluate(evaluated_data)
            {output} = evaluated_data
            """.format(algorithm=named_inputs['algorithm'],
                       input_data=named_inputs['input data'],
                       evaluator=named_inputs['evaluator'],
                       output=outputs[0],
                       folds=params[CrossValidationOperation.NUM_FOLDS_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_cross_validation_complete_operation_success():
    params = {

        CrossValidationOperation.NUM_FOLDS_PARAM: 3,

    }
    inputs = ['df_1', 'df_2', 'df_3']
    named_inputs={'algorithm': 'df_1',
                  'input data': 'df_2',
                  'evaluator': 'df_3'}
    named_outputs = {'evaluation': 'output_1'}
    outputs = ['output_1']

    instance = CrossValidationOperation(params, inputs,
                                        outputs,
                                        named_inputs=named_inputs,
                                        named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
            grid_builder = tuning.ParamGridBuilder()
            estimator, param_grid = {algorithm}

            # if estimator.__class__ == classification.LinearRegression:
            #     param_grid = estimator.maxIter
            # elif estimator.__class__  == classification.:
            #     pass
            # elif estimator.__class__ == classification.DecisionTreeClassifier:
            #     # param_grid = (estimator.maxDepth, [2,3,4,5,6,7,8,9])
            #     param_grid = (estimator.impurity, ['gini', 'entropy'])
            # elif estimator.__class__ == classification.GBTClassifier:
            #     pass
            # elif estimator.__class__ == classification.RandomForestClassifier:
            #     param_grid = estimator.maxDepth
            for param_name, values in param_grid.iteritems():
                param = getattr(estimator, param_name)
                grid_builder.addGrid(param, values)

            evaluator = {evaluator}

            cross_validator = tuning.CrossValidator(
                estimator=estimator, estimatorParamMaps=grid_builder.build(),
                evaluator=evaluator, numFolds={folds})
            cv_model = cross_validator.fit({input_data})
            evaluated_data = cv_model.transform({input_data})
            best_model_{output}  = cv_model.bestModel
            metric_result = evaluator.evaluate(evaluated_data)
            {output} = evaluated_data
            """.format(algorithm=named_inputs['algorithm'],
                       input_data=named_inputs['input data'],
                       evaluator=named_inputs['evaluator'],
                       output=outputs[0],
                       folds=params[CrossValidationOperation.NUM_FOLDS_PARAM]))

    eval_code = """
            grouped_result = evaluated_data.select(
                    evaluator.getLabelCol(), evaluator.getPredictionCol())\\
                    .groupBy(evaluator.getLabelCol(),
                             evaluator.getPredictionCol()).count().collect()
            eval_{output} = {{
                'metric': {{
                    'name': evaluator.getMetricName(),
                    'value': metric_result
                }},
                'estimator': {{
                    'name': estimator.__class__.__name__,
                    'predictionCol': evaluator.getPredictionCol(),
                    'labelCol': evaluator.getLabelCol()
                }},
                'confusion_matrix': {{
                    'data': json.dumps(grouped_result)
                }},
                'evaluator': evaluator
            }}
            """.format(output=outputs[0])
    expected_code = '\n'.join([expected_code, dedent(eval_code)])

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_cross_validation_complete_operation_missing_input_failure():
    params = {

        CrossValidationOperation.NUM_FOLDS_PARAM: 3,

    }
    inputs = ['df_1', 'df_2']
    named_inputs={'algorithm': 'df_1',
                  'input data': 'df_2',
                  'evaluator': 'df_3'}
    named_outputs = {'evaluation': 'output_1'}
    outputs = ['output_1']

    with pytest.raises(ValueError):
        CrossValidationOperation(params,
                                 inputs,
                                 outputs,
                                 named_inputs=named_inputs,
                                 named_outputs=named_outputs)


'''
    ClassificationModel tests
'''

def test_classification_model_operation_success():
    params = {
        ClassificationModel.FEATURES_ATTRIBUTE_PARAM: 'f',
        ClassificationModel.LABEL_ATTRIBUTE_PARAM: 'l'

    }
    inputs = ['df_1', 'df_2']
    outputs = ['output_1']

    instance = ClassificationModel(params, inputs,
                                        outputs,
                                        named_inputs={},
                                        named_outputs={})

    code = instance.generate_code()

    expected_code = dedent( """
        {input_2}.setLabelCol('{label}').setFeaturesCol('{features}')
        {output} = {input_2}.fit({input_1})
        """.format(output=outputs[0],
                   input_1=inputs[0],
                   input_2=inputs[1],
                   features=params[ClassificationModel.FEATURES_ATTRIBUTE_PARAM],
                   label=params[ClassificationModel.LABEL_ATTRIBUTE_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_classification_model_operation_failure():
    params = {
        ClassificationModel.FEATURES_ATTRIBUTE_PARAM: 'f',
        ClassificationModel.LABEL_ATTRIBUTE_PARAM: 'l'

    }
    inputs = ['df_1']
    outputs = ['output_1']


    with pytest.raises(ValueError):
        instance = ClassificationModel(params, inputs,
                                       outputs,
                                       named_inputs={},
                                       named_outputs={})
        instance.generate_code()


def test_classification_model_operation_missing_features_failure():
    params = {
        ClassificationModel.LABEL_ATTRIBUTE_PARAM: 'label'
    }
    inputs = ['df_1', 'df_2']
    named_inputs = {}
    named_outputs = {}
    outputs = ['output_1']

    with pytest.raises(ValueError):
        ClassificationModel(params,
                                 inputs,
                                 outputs,
                                 named_inputs=named_inputs,
                                 named_outputs=named_outputs)


def test_classification_model_operation_missing_label_failure():
    params = {
        ClassificationModel.FEATURES_ATTRIBUTE_PARAM: 'features',

    }
    inputs = ['df_1', 'df_2']
    named_inputs = {}
    named_outputs = {}
    outputs = ['output_1']

    with pytest.raises(ValueError):
        ClassificationModel(params,
                            inputs,
                            outputs,
                            named_inputs=named_inputs,
                            named_outputs=named_outputs)


def test_classification_model_operation_missing_inputs_failure():
    params = {
    }
    inputs = ['df_1', 'df_2']
    named_inputs = {}
    named_outputs = {}
    outputs = ['output_1']

    with pytest.raises(ValueError):
        classification_model = ClassificationModel(params,
                                                  inputs,
                                                  outputs,
                                                  named_inputs=named_inputs,
                                                  named_outputs=named_outputs)

        classification_model.generate_code()



'''
    ClassifierOperation tests
'''

# @FIX-ME
def test_classifier_operation_success():

    params = {
        ClassifierOperation.GRID_PARAM: {
                                         ClassifierOperation.FEATURES_PARAM: 'f',
                                         ClassifierOperation.LABEL_PARAM: 'l'
                                        }

    }
    inputs = ['df_1', 'df_2']
    outputs = ['output_1']

    instance = ClassifierOperation(params, inputs,
                                   outputs,
                                   named_inputs={},
                                   named_outputs={})

    code = instance.generate_code()

    param_grid = {
        "labelCol": params[ClassifierOperation.GRID_PARAM]
                          [ClassifierOperation.LABEL_PARAM],
        "featuresCol": params[ClassifierOperation.GRID_PARAM]
                             [ClassifierOperation.FEATURES_PARAM]
    }

    expected_code = dedent("""

    param_grid = {param_grid}
    # Output result is the classifier and its parameters. Parameters are
    # need in classification model or cross validator.
    {output} = ({name}(), param_grid)
    """.format(output=outputs[0],
               name='FIXME',
               param_grid=json.dumps(param_grid)
               )
               )


    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_classifier_operation_failure():

    params = {

    }
    inputs = ['df_1', 'df_2']
    outputs = ['output_1']

    with pytest.raises(ValueError):
        ClassifierOperation(params,
                            inputs,
                            outputs,
                            named_inputs={},
                            named_outputs={})


def test_classifier_operation_missing_label_failure():

    params = {
        ClassifierOperation.GRID_PARAM: {
                                            ClassifierOperation.FEATURES_PARAM: 'f',
                                         }

    }
    inputs = ['df_1', 'df_2']
    outputs = ['output_1']

    with pytest.raises(ValueError):
        ClassifierOperation(params,
                            inputs,
                            outputs,
                            named_inputs={},
                            named_outputs={})


def test_classifier_operation_missing_features_failure():

    params = {
        ClassifierOperation.GRID_PARAM: {
                                            ClassifierOperation.LABEL_PARAM: 'l'
                                        }
        }

    inputs = ['df_1', 'df_2']
    outputs = ['output_1']

    with pytest.raises(ValueError):
        ClassifierOperation(params,
                            inputs,
                            outputs,
                            named_inputs={},
                            named_outputs={})

def test_classifier_operation_missing_output_failure():

    params = {
        ClassifierOperation.GRID_PARAM: {
                                            ClassifierOperation.FEATURES_PARAM: 'f',
                                            ClassifierOperation.LABEL_PARAM: 'l'
                                        }

    }

    inputs = ['df_1', 'df_2']
    outputs = []


    with pytest.raises(ValueError):
        classifier = ClassifierOperation(params,
                            inputs,
                            outputs,
                            named_inputs={},
                            named_outputs={})

        classifier.generate_code()