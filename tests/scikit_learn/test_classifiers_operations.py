# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import ast
import json
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.runner import configuration
from juicer.scikit_learn.classification_operation import \
        ClassificationModelOperation, \
        DecisionTreeClassifierOperation, GBTClassifierOperation, \
        KNNClassifierOperation, LogisticRegressionOperation, \
        MLPClassifierOperation, NaiveBayesClassifierOperation, \
        PerceptronClassifierOperation, RandomForestClassifierOperation, \
        SvmClassifierOperation

from tests import compare_ast, format_code_comparison


'''
    DecisionTreeClassifierOperation Operation
'''


def test_gbt_regressor_minimum_params_success():
    params = {
    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = DecisionTreeClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        classifier_1 = DecisionTreeClassifier(max_depth=None, 
        min_samples_split=2, min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, random_state=None)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_gbt_regressor_with_params_success():
    params = {
        DecisionTreeClassifierOperation.SEED_PARAM: 14,
        DecisionTreeClassifierOperation.MIN_LEAF_PARAM: 4,
        DecisionTreeClassifierOperation.MIN_SPLIT_PARAM: 5,
        DecisionTreeClassifierOperation.MAX_DEPTH_PARAM: 11,
        DecisionTreeClassifierOperation.MIN_WEIGHT_PARAM: 0.1
    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = DecisionTreeClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        classifier_1 = DecisionTreeClassifier(max_depth=11, 
        min_samples_split=5, min_samples_leaf=4, 
        min_weight_fraction_leaf=0.1, random_state=14)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_gbt_regressor_wrong_value_param_failure():
    params = {
        DecisionTreeClassifierOperation.MAX_DEPTH_PARAM: -10
    }
    n_out = {'algorithm': 'classifier_1'}
    with pytest.raises(ValueError):
        DecisionTreeClassifierOperation(params, named_inputs={},
                                        named_outputs=n_out)


'''
    GBT Classifier Operation
'''


def test_gbt_classifier_minimum_params_success():
    params = {
    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = GBTClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        classifier_1 = GradientBoostingClassifier(loss='deviance',
          learning_rate=0.1, n_estimators=100, 
          min_samples_split=2, max_depth=3,
          min_samples_leaf=1, random_state=None)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_gbt_classifier_with_params_success():
    params = {
        GBTClassifierOperation.N_ESTIMATORS_PARAM: 11,
        GBTClassifierOperation.MIN_LEAF_PARAM: 10,
        GBTClassifierOperation.MIN_SPLIT_PARAM: 12,
        GBTClassifierOperation.LEARNING_RATE_PARAM: 1.1,
        GBTClassifierOperation.MAX_DEPTH_PARAM: 13,
        GBTClassifierOperation.SEED_PARAM: 9,
        GBTClassifierOperation.LOSS_PARAM:
            GBTClassifierOperation.LOSS_PARAM_EXP
    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = GBTClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        classifier_1 = GradientBoostingClassifier(loss='exponencial',
          learning_rate=1.1, n_estimators=11,
          min_samples_split=12, max_depth=13,
          min_samples_leaf=10, random_state=9)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_gbt_classifier_wrong_value_param_failure():
    params = {
        GBTClassifierOperation.N_ESTIMATORS_PARAM: -10
    }
    n_out = {'algorithm': 'classifier_1'}
    with pytest.raises(ValueError):
        GBTClassifierOperation(params, named_inputs={}, named_outputs=n_out)


'''
   KNN Classifier Operation
'''


def test_knn_classifier_minimum_params_success():
    params = {
        KNNClassifierOperation.K_PARAM: 3
    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = KNNClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        classifier_1 = KNeighborsClassifier(n_neighbors=3)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_knn_classifier_wrong_value_param_failure():
    params = {
        KNNClassifierOperation.K_PARAM: -3
    }
    n_out = {'algorithm': 'classifier_1'}
    with pytest.raises(ValueError):
        KNNClassifierOperation(params, named_inputs={}, named_outputs=n_out)


'''
    Logistic Regression Operation
'''


def test_logisticregression_minimum_params_success():
    params = {
    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = LogisticRegressionOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        classifier_1 = LogisticRegression(tol=0.0001, C=1.0, max_iter=100, 
        solver='liblinear', random_state=None)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_logisticregression_with_params_success():
    params = {
        LogisticRegressionOperation.TOLERANCE_PARAM: 0.1,
        LogisticRegressionOperation.MAX_ITER_PARAM: 10,
        LogisticRegressionOperation.SEED_PARAM: 2,
        LogisticRegressionOperation.REGULARIZATION_PARAM: 1.1,
        LogisticRegressionOperation.SOLVER_PARAM:
            LogisticRegressionOperation.SOLVER_PARAM_NEWTON
    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = LogisticRegressionOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        classifier_1 = LogisticRegression(tol=0.1, C=1.1, max_iter=10, 
        solver='newton-cg', random_state=2)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_logisticregression_wrong_value_param_failure():
    params = {
        LogisticRegressionOperation.MAX_ITER_PARAM: -1.0
    }
    n_in = {}
    n_out = {'algorithm': 'classifier_1'}
    with pytest.raises(ValueError):
        LogisticRegressionOperation(params, named_inputs=n_in,
                                    named_outputs=n_out)


'''
    MLP Classifier Operation
'''


def test_mlp_classifier_minimum_params_success():
    params = {
        MLPClassifierOperation.HIDDEN_LAYER_SIZES_PARAM: '(100,100,9)'
    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = MLPClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        classifier_1 = MLPClassifier(hidden_layer_sizes=(100,100,9),
        activation='relu', solver='adam', alpha=0.0001,
        max_iter=200, random_state=None, tol=0.0001)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_mlp_classifier_with_params_success():
    params = {
        MLPClassifierOperation.HIDDEN_LAYER_SIZES_PARAM: '(100,10,9)',
        MLPClassifierOperation.ACTIVATION_PARAM:
            MLPClassifierOperation.ACTIVATION_PARAM_LOG,
        MLPClassifierOperation.SEED_PARAM: 9,
        MLPClassifierOperation.SOLVER_PARAM:
            MLPClassifierOperation.SOLVER_PARAM_LBFGS,
        MLPClassifierOperation.MAX_ITER_PARAM: 1000,
        MLPClassifierOperation.ALPHA_PARAM: 0.01,
        MLPClassifierOperation.TOLERANCE_PARAM: 0.1,
    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = MLPClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        classifier_1 = MLPClassifier(hidden_layer_sizes=(100,10,9),
        activation='logistic', solver='lbfgs', alpha=0.01,
        max_iter=1000, random_state=9, tol=0.1)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_mlp_classifier_wrong_value_param_failure():
    params = {
        MLPClassifierOperation.HIDDEN_LAYER_SIZES_PARAM: '100.100,'
    }
    n_out = {'algorithm': 'classifier_1'}
    with pytest.raises(ValueError):
        MLPClassifierOperation(params, named_inputs={}, named_outputs=n_out)


'''
  Naive Bayes Operation
'''


def test_naive_bayes_minimum_params_success():
    params = {
    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = NaiveBayesClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        classifier_1 = MultinomialNB(alpha=1.0, prior=None)""")

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_naive_bayes_with_params_success():
    params = {

        NaiveBayesClassifierOperation.ALPHA_PARAM: 2.0,
        NaiveBayesClassifierOperation.MODEL_TYPE_PARAM:
            NaiveBayesClassifierOperation.MODEL_TYPE_PARAM_B,
        NaiveBayesClassifierOperation.CLASS_PRIOR_PARAM: '1,2,3,4,5',
    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = NaiveBayesClassifierOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        classifier_1 = BernoulliNB(alpha=2.0, prior=[1,2,3,4,5])""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_naive_bayes_wrong_value_param_failure():
    params = {
        NaiveBayesClassifierOperation.ALPHA_PARAM: -1
    }
    n_in = {}
    n_out = {'algorithm': 'classifier_1'}
    with pytest.raises(ValueError):
        NaiveBayesClassifierOperation(params, named_inputs=n_in,
                                      named_outputs=n_out)


'''
    Perceptron Classifier Operation
'''


def test_perceptron_minimum_params_success():
    params = {
    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = PerceptronClassifierOperation(params, named_inputs={},
                                                named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        classifier_1 = Perceptron(tol=0.001, alpha=0.0001, max_iter=1000,
        shuffle=False, random_state=None, penalty='None')""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_perceptron_with_params_success():
    params = {
        PerceptronClassifierOperation.SHUFFLE_PARAM: True,
        PerceptronClassifierOperation.PENALTY_PARAM:
            PerceptronClassifierOperation.PENALTY_PARAM_EN,
        PerceptronClassifierOperation.SEED_PARAM: 10,
        PerceptronClassifierOperation.ALPHA_PARAM: 0.11,
        PerceptronClassifierOperation.TOLERANCE_PARAM: 0.1,
        PerceptronClassifierOperation.MAX_ITER_PARAM: 100
    }
    n_out = {'algorithm': 'classifier_1'}

    instance_lr = PerceptronClassifierOperation(params, named_inputs={},
                                                named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        classifier_1 = Perceptron(tol=0.1, alpha=0.11, max_iter=100,
        shuffle=True, random_state=10, penalty='elasticnet')""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_perceptron_wrong_value_param_failure():
    params = {
        PerceptronClassifierOperation.ALPHA_PARAM: -1.0
    }
    n_in = {}
    n_out = {'algorithm': 'classifier_1'}
    with pytest.raises(ValueError):
        PerceptronClassifierOperation(params, named_inputs=n_in,
                                      named_outputs=n_out)


'''
  RandomForestClassifierOperation
'''


def test_random_forest_operation_minimum_success():
    params = {
    }
    n_in = {}
    n_out = {'algorithm': 'classifier_1'}

    instance = RandomForestClassifierOperation(params, named_inputs=n_in,
                                               named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        classifier_1 = RandomForestClassifier(n_estimators=10, 
         max_depth=None,  min_samples_split=2, 
         min_samples_leaf=1, random_state=None)
    """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_random_forest_operation_params_success():
    params = {
        RandomForestClassifierOperation.SEED_PARAM: 10,
        RandomForestClassifierOperation.MAX_DEPTH_PARAM: 11,
        RandomForestClassifierOperation.MIN_SPLIT_PARAM: 12,
        RandomForestClassifierOperation.MIN_LEAF_PARAM: 13,
        RandomForestClassifierOperation.N_ESTIMATORS_PARAM: 15
    }
    n_in = {}
    n_out = {'algorithm': 'classifier_1'}

    instance = RandomForestClassifierOperation(params, named_inputs=n_in,
                                               named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        classifier_1 = RandomForestClassifier(n_estimators=15, 
         max_depth=11,  min_samples_split=12, 
         min_samples_leaf=13, random_state=10)
    """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_random_forest_wrong_value_param_failure():
    params = {
        RandomForestClassifierOperation.N_ESTIMATORS_PARAM: -1
    }
    n_in = {}
    n_out = {'algorithm': 'classifier_1'}
    with pytest.raises(ValueError):
        RandomForestClassifierOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)


'''
    SVM Classifier Operation
'''


def test_svm_operation_minimum_success():
    params = {
    }
    n_in = {}
    n_out = {'algorithm': 'classifier_1'}

    instance = SvmClassifierOperation(params, named_inputs=n_in,
                                      named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        classifier_1 = SVC(tol=0.001, C=1.0, max_iter=-1, 
                           degree=3, kernel='rbf', random_state=None)
    """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_svm_operation_params_success():
    params = {
        SvmClassifierOperation.PENALTY_PARAM: 10.0,
        SvmClassifierOperation.KERNEL_PARAM:
            SvmClassifierOperation.KERNEL_PARAM_POLY,
        SvmClassifierOperation.DEGREE_PARAM: 2,
        SvmClassifierOperation.TOLERANCE_PARAM: -0.1,
        SvmClassifierOperation.MAX_ITER_PARAM: 13,
        SvmClassifierOperation.SEED_PARAM: 12
    }
    n_in = {}
    n_out = {'algorithm': 'classifier_1'}

    instance = SvmClassifierOperation(params, named_inputs=n_in,
                                      named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        classifier_1 = SVC(tol=0.1, C=10.0, max_iter=13, 
                           degree=2, kernel='poly', random_state=12)
    """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_svm_wrong_value_param_failure():
    params = {
        SvmClassifierOperation.DEGREE_PARAM: -1
    }
    n_in = {}
    n_out = {'algorithm': 'classifier_1'}
    with pytest.raises(ValueError):
        SvmClassifierOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    Classification Model Operation    
'''


def test_classification_operation_success():
    params = {

        ClassificationModelOperation.FEATURES_ATTRIBUTE_PARAM: ['f'],
        ClassificationModelOperation.LABEL_ATTRIBUTE_PARAM: ['label']

    }
    named_inputs = {'algorithm': 'algo',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1'}

    instance = ClassificationModelOperation(params, named_inputs=named_inputs,
                                            named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        X = df_2['f'].values.tolist()
        y = df_2['label'].values.tolist()
        model_task_1 = algo.fit(X, y)
        
        output_1 = df_2
         
        output_1['prediction'] = model_task_1.predict(X).tolist()
        """.format())

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_classification_with_model_operation_success():
    params = {

        ClassificationModelOperation.FEATURES_ATTRIBUTE_PARAM: ['f'],
        ClassificationModelOperation.LABEL_ATTRIBUTE_PARAM: ['label']

    }
    named_inputs = {'algorithm': 'algo',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}

    instance = ClassificationModelOperation(params, named_inputs=named_inputs,
                                            named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        X = df_2['f'].values.tolist()
        y = df_2['label'].values.tolist()
        output_2 = algo.fit(X, y)
         
        output_1 = df_2
         
        output_1['prediction'] = output_2.predict(X).tolist()
        """.format())

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_classification_model_operation_success():
    params = {

        ClassificationModelOperation.FEATURES_ATTRIBUTE_PARAM: ['f'],
        ClassificationModelOperation.LABEL_ATTRIBUTE_PARAM: ['label']

    }
    named_inputs = {'algorithm': 'algo',
                    'train input data': 'df_2'}
    named_outputs = {'model': 'output_2'}

    instance = ClassificationModelOperation(params, named_inputs=named_inputs,
                                            named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        X = df_2['f'].values.tolist()
        y = df_2['label'].values.tolist()
        output_2 = algo.fit(X, y)

        task_1 = None
        """.format())

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_classification_model_operation_missing_features_failure():
    params = {}
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}

    with pytest.raises(ValueError):
        ClassificationModelOperation(params, named_inputs=named_inputs,
                                     named_outputs=named_outputs)


def test_classification_model_operation_missing_input_failure():
    params = {
        ClassificationModelOperation.FEATURES_ATTRIBUTE_PARAM: ['f']
    }
    named_inputs = {'algorithm': 'df_1'}
    named_outputs = {'output data': 'output_1'}

    with pytest.raises(ValueError):
        ClassificationModelOperation(params, named_inputs=named_inputs,
                                     named_outputs=named_outputs)


def test_classification_model_operation_missing_output_success():
    params = {
        ClassificationModelOperation.FEATURES_ATTRIBUTE_PARAM: ['f'],
        ClassificationModelOperation.LABEL_ATTRIBUTE_PARAM: ['label']
    }
    named_inputs = {'algorithm': 'df_1', 'train input data': 'df_2'}
    named_outputs = {'model': 'output_2'}

    classifier = ClassificationModelOperation(params,
                                              named_inputs=named_inputs,
                                              named_outputs=named_outputs)
    assert classifier.has_code
