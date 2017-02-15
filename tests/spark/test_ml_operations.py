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
    ClassifierOperation, SvmClassifierOperation, \
    DecisionTreeClassifierOperation, \
    GBTClassifierOperation, \
    NaiveBayesClassifierOperation, \
    RandomForestClassifierOperation, \
    LogisticRegressionClassifierOperation, \
    PerceptronClassifier, \
    ClusteringOperation, ClusteringModelOperation, \
    LdaClusteringOperation, KMeansClusteringOperation, \
    GaussianMixtureClusteringOperation, TopicReportOperation, \
    CollaborativeOperation, AlternatingLeastSquaresOperation

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
                   input_1=inputs[1],
                   input_2=inputs[0]))

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
    named_inputs={'input data':'input_1',
                   'model' :'df_model'}
    named_outputs={'metric' :'df_metric',
                   'evaluator':'df_evaluator'}

    instance = EvaluateModel(params, inputs,
                             outputs, named_inputs=named_inputs,
                             named_outputs=named_outputs)

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
    inputs = ['df_1', 'df_2', 'df_3']
    named_inputs = {'algorithm': 'df_1',
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
    named_inputs = {'algorithm': 'df_1',
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
    named_inputs = {'algorithm': 'df_1',
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

    expected_code = dedent("""
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


def test_svm_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    inputs = ['df_1', 'df_2']
    outputs = ['output_1']

    instance_svm = SvmClassifierOperation(params, inputs,
                                          outputs,
                                          named_inputs={},
                                          named_outputs={})

    # Is not possible to generate_code(), because has_code is False
    assert instance_svm.name == 'classification.SVM'


def test_lr_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    inputs = ['df_1', 'df_2']
    outputs = ['output_1']

    instance_lr = LogisticRegressionClassifierOperation(params, inputs,
                                                        outputs,
                                                        named_inputs={},
                                                        named_outputs={})

    # Is not possible to generate_code(), because has_code is False
    assert instance_lr.name == 'classification.LogisticRegression'


def test_dt_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    inputs = ['df_1', 'df_2']
    outputs = ['output_1']

    instance_dt = DecisionTreeClassifierOperation(params, inputs,
                                                  outputs,
                                                  named_inputs={},
                                                  named_outputs={})

    # Is not possible to generate_code(), because has_code is False
    assert instance_dt.name == 'classification.DecisionTreeClassifier'


def test_gbt_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    inputs = ['df_1', 'df_2']
    outputs = ['output_1']

    instance_dt = GBTClassifierOperation(params, inputs,
                                         outputs,
                                         named_inputs={},
                                         named_outputs={})

    # Is not possible to generate_code(), because has_code is False
    assert instance_dt.name == 'classification.GBTClassifier'


def test_nb_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    inputs = ['df_1', 'df_2']
    outputs = ['output_1']

    instance_nb = NaiveBayesClassifierOperation(params, inputs,
                                                outputs,
                                                named_inputs={},
                                                named_outputs={})

    # Is not possible to generate_code(), because has_code is False
    assert instance_nb.name == 'classification.NaiveBayes'


def test_rf_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    inputs = ['df_1', 'df_2']
    outputs = ['output_1']

    instance_nb = RandomForestClassifierOperation(params, inputs,
                                                  outputs,
                                                  named_inputs={},
                                                  named_outputs={})

    # Is not possible to generate_code(), because has_code is False
    assert instance_nb.name == 'classification.RandomForestClassifier'


def test_percept_classifier_operation_success():
    params = {
        ClassifierOperation.GRID_PARAM: {
            ClassifierOperation.FEATURES_PARAM: 'f',
            ClassifierOperation.LABEL_PARAM: 'l'
        }

    }
    inputs = ['df_1', 'df_2']
    outputs = ['output_1']

    instance_pct = PerceptronClassifier(params, inputs,
                                        outputs,
                                        named_inputs={},
                                        named_outputs={})

    # Is not possible to generate_code(), because has_code is False
    assert instance_pct.name == 'classification.MultilayerPerceptronClassificationModel'


"""
    Clustering tests
"""


def test_clustering_model_operation_success():
    params = {

        ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM: 'f',

    }
    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = ['output_1']

    instance = ClusteringModelOperation(params, inputs,
                                        outputs,
                                        named_inputs=named_inputs,
                                        named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        {algorithm}.setFeaturesCol('{features}')
        {model} = {algorithm}.fit({input})
        # There is no way to pass which attribute was used in clustering, so
        # this information will be stored in uid (hack).
        {model}.uid += '|{features}'
        {output} = {model}.transform({input})
        """.format(algorithm=named_inputs['algorithm'],
                   input=named_inputs['train input data'],
                   model=named_outputs['model'],
                   output=outputs[0],
                   features=params[ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_clustering_model_operation_missing_features_failure():
    params = {
    }
    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = ['output_1']

    with pytest.raises(ValueError):
        ClusteringModelOperation(params,
                                 inputs,
                                 outputs,
                                 named_inputs=named_inputs,
                                 named_outputs=named_outputs)


def test_clustering_model_operation_missing_input_failure():
    params = {

        ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM: 'f',

    }
    inputs = ['df_1']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = ['output_1']

    with pytest.raises(ValueError):
        clustering = ClusteringModelOperation(params,
                                              inputs,
                                              outputs,
                                              named_inputs=named_inputs,
                                              named_outputs=named_outputs)
        clustering.generate_code()


def test_clustering_model_operation_missing_output_failure():
    params = {

        ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM: 'f',

    }
    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = []

    with pytest.raises(ValueError):
        clustering = ClusteringModelOperation(params,
                                              inputs,
                                              outputs,
                                              named_inputs=named_inputs,
                                              named_outputs=named_outputs)
        clustering.generate_code()


def test_clustering_operation_success():
    # This test its not very clear, @CHECK
    params = {}
    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = ['output_1']

    name = 'FIXME'
    set_values = []
    instance = ClusteringOperation(params, inputs,
                                   outputs,
                                   named_inputs=named_inputs,
                                   named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=outputs[0],
        name=name))

    settings = (['{0}.set{1}({2})'.format(outputs[0], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_lda_clustering_operation_optimizer_online_success():
    params = {
        LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM: 10,
        LdaClusteringOperation.OPTIMIZER_PARAM: 'online',
        LdaClusteringOperation.MAX_ITERATIONS_PARAM: 10,
        LdaClusteringOperation.DOC_CONCENTRATION_PARAM: 0.25,
        LdaClusteringOperation.TOPIC_CONCENTRATION_PARAM: 0.1
    }
    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = ['output_1']

    name = "clustering.LDA"

    set_values = [
        ['DocConcentration', params[LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM] *
         [(params.get(LdaClusteringOperation.DOC_CONCENTRATION_PARAM,
                      LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM))
          / 50.0]],
        ['K', params[LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM]],
        ['MaxIter', params[LdaClusteringOperation.MAX_ITERATIONS_PARAM]],
        ['Optimizer', "'{}'".format(params[LdaClusteringOperation.OPTIMIZER_PARAM])],
        ['TopicConcentration', params[LdaClusteringOperation.TOPIC_CONCENTRATION_PARAM]]
    ]

    instance = LdaClusteringOperation(params, inputs,
                                      outputs,
                                      named_inputs=named_inputs,
                                      named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=outputs[0],
        name=name))

    settings = (['{0}.set{1}({2})'.format(outputs[0], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


# def test_lda_clustering_operation_optimizer_em_success():
#     params = {
#         LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM: 10,
#         LdaClusteringOperation.OPTIMIZER_PARAM: 'em',
#         LdaClusteringOperation.MAX_ITERATIONS_PARAM: 10,
#         LdaClusteringOperation.DOC_CONCENTRATION_PARAM: 0.25,
#         LdaClusteringOperation.TOPIC_CONCENTRATION_PARAM: 0.1
#         LdaClusteringOperation.ONLINE_OPTIMIZER: '',
#         LdaClusteringOperation.EM_OPTIMIZER: ''
#
# }
# inputs = ['df_1', 'df_2']
# outputs = ['output_1']
#
# instance = LdaClusteringOperation(params, inputs,
#                                   outputs,
#                                   named_inputs={},
#                                   named_outputs={})
#
# code = instance.generate_code()
#
# expected_code = dedent("""
#     {input_2}.setLabelCol('{label}').setFeaturesCol('{features}')
#     {output} = {input_2}.fit({input_1})
#     """.format(output=outputs[0],
#                input_1=inputs[0],
#                input_2=inputs[1],
#                features=params[ClassificationModel.FEATURES_ATTRIBUTE_PARAM],
#                label=params[ClassificationModel.LABEL_ATTRIBUTE_PARAM]))
#
# result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#
# assert result, msg + debug_ast(code, expected_code)


def test_lda_clustering_operation_failure():
    params = {
        LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM: 10,
        LdaClusteringOperation.OPTIMIZER_PARAM: 'xXx',
        LdaClusteringOperation.MAX_ITERATIONS_PARAM: 10,
        LdaClusteringOperation.DOC_CONCENTRATION_PARAM: 0.25,
        LdaClusteringOperation.TOPIC_CONCENTRATION_PARAM: 0.1
    }
    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = ['output_1']
    name = 'FIXME'

    with pytest.raises(ValueError):
        LdaClusteringOperation(params, inputs,
                               outputs,
                               named_inputs=named_inputs,
                               named_outputs=named_outputs)


def test_kmeans_clustering_operation_random_type_kmeans_success():
    params = {
        KMeansClusteringOperation.K_PARAM: 10,
        KMeansClusteringOperation.MAX_ITERATIONS_PARAM: 20,
        KMeansClusteringOperation.TYPE_PARAMETER: 'kmeans',
        KMeansClusteringOperation.INIT_MODE_PARAMETER: 'random',
        KMeansClusteringOperation.TOLERANCE_PARAMETER: 0.001
    }
    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = ['output_1']

    name = "clustering.KMeans"

    set_values = [
        ['MaxIter', params[KMeansClusteringOperation.MAX_ITERATIONS_PARAM]],
        ['K', params[KMeansClusteringOperation.K_PARAM]],
        ['Tol', params[KMeansClusteringOperation.TOLERANCE_PARAMETER]],
        ['InitMode',
         '"{}"'.format(params[KMeansClusteringOperation.INIT_MODE_PARAMETER])]
    ]

    instance = KMeansClusteringOperation(params, inputs,
                                         outputs,
                                         named_inputs=named_inputs,
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=outputs[0],
        name=name))

    settings = (['{0}.set{1}({2})'.format(outputs[0], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_kmeans_clustering_operation_random_type_bisecting_success():
    params = {
        KMeansClusteringOperation.K_PARAM: 10,
        KMeansClusteringOperation.MAX_ITERATIONS_PARAM: 20,
        KMeansClusteringOperation.TYPE_PARAMETER: 'bisecting',
        KMeansClusteringOperation.INIT_MODE_PARAMETER: 'random',
        KMeansClusteringOperation.TOLERANCE_PARAMETER: 0.001
    }
    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = ['output_1']

    name = "BisectingKMeans"

    set_values = [
        ['MaxIter', params[KMeansClusteringOperation.MAX_ITERATIONS_PARAM]],
        ['K', params[KMeansClusteringOperation.K_PARAM]],
        ['Tol', params[KMeansClusteringOperation.TOLERANCE_PARAMETER]],
        # ['InitMode', params[KMeansClusteringOperation.INIT_MODE_PARAMETER]]
    ]

    instance = KMeansClusteringOperation(params, inputs,
                                         outputs,
                                         named_inputs=named_inputs,
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=outputs[0],
        name=name))

    settings = (['{0}.set{1}({2})'.format(outputs[0], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_kmeans_clustering_operation_kmeansdd_type_kmeans_success():
    params = {
        KMeansClusteringOperation.K_PARAM: 10,
        KMeansClusteringOperation.MAX_ITERATIONS_PARAM: 20,
        KMeansClusteringOperation.TYPE_PARAMETER: 'kmeans',
        KMeansClusteringOperation.INIT_MODE_PARAMETER: 'k-means||',
        KMeansClusteringOperation.TOLERANCE_PARAMETER: 0.001
    }
    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = ['output_1']

    name = "clustering.KMeans"

    set_values = [
        ['MaxIter', params[KMeansClusteringOperation.MAX_ITERATIONS_PARAM]],
        ['K', params[KMeansClusteringOperation.K_PARAM]],
        ['Tol', params[KMeansClusteringOperation.TOLERANCE_PARAMETER]],
        ['InitMode',
         '"{}"'.format(params[KMeansClusteringOperation.INIT_MODE_PARAMETER])]
    ]

    instance = KMeansClusteringOperation(params, inputs,
                                         outputs,
                                         named_inputs=named_inputs,
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=outputs[0],
        name=name))

    settings = (['{0}.set{1}({2})'.format(outputs[0], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_kmeans_clustering_operation_kmeansdd_type_bisecting_success():
    params = {
        KMeansClusteringOperation.K_PARAM: 10,
        KMeansClusteringOperation.MAX_ITERATIONS_PARAM: 20,
        KMeansClusteringOperation.TYPE_PARAMETER: 'bisecting',
        KMeansClusteringOperation.INIT_MODE_PARAMETER: 'k-means||',
        KMeansClusteringOperation.TOLERANCE_PARAMETER: 0.001
    }
    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = ['output_1']

    name = "BisectingKMeans"

    set_values = [
        ['MaxIter', params[KMeansClusteringOperation.MAX_ITERATIONS_PARAM]],
        ['K', params[KMeansClusteringOperation.K_PARAM]],
        ['Tol', params[KMeansClusteringOperation.TOLERANCE_PARAMETER]],
        # ['InitMode', params[KMeansClusteringOperation.INIT_MODE_PARAMETER]]
    ]

    instance = KMeansClusteringOperation(params, inputs,
                                         outputs,
                                         named_inputs=named_inputs,
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=outputs[0],
        name=name))

    settings = (['{0}.set{1}({2})'.format(outputs[0], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_kmeans_clustering_operation_random_type_failure():
    params = {
        KMeansClusteringOperation.K_PARAM: 10,
        KMeansClusteringOperation.MAX_ITERATIONS_PARAM: 20,
        KMeansClusteringOperation.TYPE_PARAMETER: 'XxX',
        KMeansClusteringOperation.INIT_MODE_PARAMETER: 'random',
        KMeansClusteringOperation.TOLERANCE_PARAMETER: 0.001
    }
    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = ['output_1']

    with pytest.raises(ValueError):
        KMeansClusteringOperation(params,
                                  inputs,
                                  outputs,
                                  named_inputs=named_inputs,
                                  named_outputs=named_outputs)


def test_gaussian_mixture_clustering_operation_success():
    params = {
        GaussianMixtureClusteringOperation.K_PARAM: 10,
        GaussianMixtureClusteringOperation.MAX_ITERATIONS_PARAM: 10,
        GaussianMixtureClusteringOperation.TOLERANCE_PARAMETER: 0.001
    }
    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}
    outputs = ['output_1']

    name = "clustering.GaussianMixture"

    set_values = [
        ['MaxIter', params[GaussianMixtureClusteringOperation.MAX_ITERATIONS_PARAM]],
        ['K', params[GaussianMixtureClusteringOperation.K_PARAM]],
        ['Tol', params[GaussianMixtureClusteringOperation.TOLERANCE_PARAMETER]],
    ]

    instance = GaussianMixtureClusteringOperation(params, inputs,
                                                  outputs,
                                                  named_inputs=named_inputs,
                                                  named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=outputs[0],
        name=name))

    settings = (['{0}.set{1}({2})'.format(outputs[0], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_topics_report_operation_success():
    params = {
        TopicReportOperation.TERMS_PER_TOPIC_PARAM: 20,
    }
    inputs = ['df_1', 'df_2']
    named_inputs = {'model': 'df_1',
                    'input data': 'df_2',
                    'vocabulary': 'df_3'}
    named_outputs = {'output data': 'output_1'}
    outputs = ['output_1']

    instance = TopicReportOperation(params, inputs,
                                    outputs,
                                    named_inputs=named_inputs,
                                    named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
            topic_df = {model}.describeTopics(maxTermsPerTopic={tpt})
            # See hack in ClusteringModelOperation
            features = {model}.uid.split('|')[1]
            '''
            for row in topic_df.collect():
                topic_number = row[0]
                topic_terms  = row[1]
                print "Topic: ", topic_number
                print '========================='
                print '\\t',
                for inx in topic_terms[:{tpt}]:
                    print {vocabulary}[features][inx],
                print
            '''
            {output} =  {input}
        """.format(model=named_inputs['model'],
                   tpt=params[TopicReportOperation.TERMS_PER_TOPIC_PARAM],
                   vocabulary=named_inputs['vocabulary'],
                   output=named_outputs['output data'],
                   input=named_inputs['input data']))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


"""
  Collaborative Filtering tests
"""


def test_collaborative_filtering_operation_success():
    params = {
    }
    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2',
                    'vocabulary': 'df_3'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_1_model'}
    outputs = ['output_1']

    name = "als"
    set_values = []

    instance = CollaborativeOperation(params, inputs,
                                      outputs,
                                      named_inputs=named_inputs,
                                      named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("{output} = {name}()".format(
        output=outputs[0],
        name=name))

    settings = (['{0}.set{1}({2})'.format(outputs[0], name, v)
                 for name, v in set_values])
    settings = "\n".join(settings)

    expected_code += "\n" + settings

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)


def test_als_operation_success():
    params = {
        AlternatingLeastSquaresOperation.RANK_PARAM: 10,
        AlternatingLeastSquaresOperation.MAX_ITER_PARAM: 10,
        AlternatingLeastSquaresOperation.USER_COL_PARAM: 'u',
        AlternatingLeastSquaresOperation.ITEM_COL_PARAM: 'm',
        AlternatingLeastSquaresOperation.RATING_COL_PARAM: 'r',
        AlternatingLeastSquaresOperation.REG_PARAM: 0.1,
        AlternatingLeastSquaresOperation.IMPLICIT_PREFS_PARAM: False,

        # Could be required
        # AlternatingLeastSquaresOperation.ALPHA_PARAM:'alpha',
        # AlternatingLeastSquaresOperation.SEED_PARAM:'seed',
        # AlternatingLeastSquaresOperation.NUM_USER_BLOCKS_PARAM:'numUserBlocks',
        # AlternatingLeastSquaresOperation.NUM_ITEM_BLOCKS_PARAM:'numItemBlocks',
    }

    inputs = ['df_1', 'df_2']
    named_inputs = {'algorithm': 'df_1',
                    'input data': 'df_2'}
    named_outputs = {'algorithm': 'algorithm_als'}
    outputs = ['output_1']

    name = "collaborativefiltering.ALS"

    instance = AlternatingLeastSquaresOperation(params, inputs,
                                                outputs,
                                                named_inputs=named_inputs,
                                                named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
                # Build the recommendation model using ALS on the training data
                {output} = ALS(maxIter={maxIter}, regParam={regParam},
                        userCol='{userCol}', itemCol='{itemCol}',
                        ratingCol='{ratingCol}')

                ##model = als.fit({input})
                #predictions = model.transform(test)

                # Evaluate the model not support YET
                # evaluator = RegressionEvaluator(metricName="rmse",
                #                labelCol={ratingCol},
                #                predictionCol="prediction")

                # rmse = evaluator.evaluate(predictions)
                # print("Root-mean-square error = " + str(rmse))
                """.format(
        output=named_outputs['algorithm'],
        input=inputs[0],
        maxIter=params[AlternatingLeastSquaresOperation.MAX_ITER_PARAM],
        regParam=params[AlternatingLeastSquaresOperation.REG_PARAM],
        userCol=params[AlternatingLeastSquaresOperation.USER_COL_PARAM],
        itemCol=params[AlternatingLeastSquaresOperation.ITEM_COL_PARAM],
        ratingCol=params[AlternatingLeastSquaresOperation.RATING_COL_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + debug_ast(code, expected_code)
