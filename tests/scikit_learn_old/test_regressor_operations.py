# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import ast
import json
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.runner import configuration
from juicer.scikit_learn.regression_operation import RegressionModelOperation, \
    GradientBoostingRegressorOperation, \
    HuberRegressorOperation, \
    IsotonicRegressionOperation, \
    LinearRegressionOperation, \
    MLPRegressorOperation, \
    RandomForestRegressorOperation, \
    SGDRegressorOperation

from tests import compare_ast, format_code_comparison


'''
    Gradient Boosting Regressor Operation
'''


def test_gbt_regressor_minimum_params_success():
    params = {
    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = GradientBoostingRegressorOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = GradientBoostingRegressor(learning_rate=0.1,
            n_estimators=100, max_depth=3, min_samples_split=2, 
            min_samples_leaf=1, random_state=None)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_gbt_regressor_with_params_success():
    params = {
        GradientBoostingRegressorOperation.N_ESTIMATORS_PARAM: 11,
        GradientBoostingRegressorOperation.MIN_SPLIT_PARAM: 12,
        GradientBoostingRegressorOperation.SEED_PARAM: 13,
        GradientBoostingRegressorOperation.MAX_DEPTH_PARAM: 14,
        GradientBoostingRegressorOperation.LEARNING_RATE_PARAM: 0.155,
        GradientBoostingRegressorOperation.MIN_LEAF_PARAM:16
    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = GradientBoostingRegressorOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = GradientBoostingRegressor(learning_rate=0.155,
          n_estimators=11, max_depth=14, min_samples_split=12, 
          min_samples_leaf=16, random_state=13)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_gbt_regressor_wrong_value_param_failure():
    params = {
        GradientBoostingRegressorOperation.N_ESTIMATORS_PARAM: -10
    }
    n_in = {}
    n_out = {'algorithm': 'regressor_1'}
    with pytest.raises(ValueError):
        GradientBoostingRegressorOperation(params, named_inputs=n_in,
                                           named_outputs=n_out)


'''
    Huber Regressor Operation
'''


def test_huber_regressor_minimum_params_success():
    params = {
    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = HuberRegressorOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001,
                                     tol=1e-05)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_huber_regressor_with_params_success():
    params = {
        HuberRegressorOperation.MAX_ITER_PARAM: 11,
        HuberRegressorOperation.TOLERANCE_PARAM: 0.1,
        HuberRegressorOperation.ALPHA_PARAM: 0.11,
        HuberRegressorOperation.EPSILON_PARAM: 1.6
    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = HuberRegressorOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = HuberRegressor(epsilon=1.6, max_iter=11, alpha=0.11,
                                     tol=1e-01)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_huber_regressor_wrong_value_param_failure():
    params = {
        HuberRegressorOperation.EPSILON_PARAM: 1.0
    }
    n_in = {}
    n_out = {'algorithm': 'regressor_1'}
    with pytest.raises(ValueError):
        HuberRegressorOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    Isotonic Regressor Operation
'''


def test_isotonic_regressor_minimum_params_success():
    params = {
    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = IsotonicRegressionOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = IsotonicRegression(increasing=True)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_isotonic_regressor_with_params_success():
    params = {
        IsotonicRegressionOperation.ISOTONIC_PARAM: False
    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = IsotonicRegressionOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = IsotonicRegression(increasing=False)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


'''
    Linear Regression Operation
'''


def test_linearegression_minimum_params_success():
    params = {
    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = LinearRegressionOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = ElasticNet(alpha=1.0, l1_ratio=0.5, tol=0.0001,
                                 max_iter=1000, random_state=None,
                                 normalize=True)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_linearegression_with_params_success():
    params = {
        LinearRegressionOperation.NORMALIZE_PARAM: False,
        LinearRegressionOperation.ALPHA_PARAM: 0.5,
        LinearRegressionOperation.ELASTIC_NET_PARAM: 0.55,
        LinearRegressionOperation.TOLERANCE_PARAM: 0.1,
        LinearRegressionOperation.MAX_ITER_PARAM: 10,
        LinearRegressionOperation.SEED_PARAM: 2
    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = LinearRegressionOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = ElasticNet(alpha=0.5, l1_ratio=0.55, tol=0.1,
                                 max_iter=10, random_state=2,
                                 normalize=False)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_linearegression_wrong_value_param_failure():
    params = {
        LinearRegressionOperation.MAX_ITER_PARAM: -1.0
    }
    n_in = {}
    n_out = {'algorithm': 'regressor_1'}
    with pytest.raises(ValueError):
        LinearRegressionOperation(params, named_inputs=n_in,
                                  named_outputs=n_out)


'''
    MLP Regressor Operation
'''


def test_mlp_regressor_minimum_params_success():
    params = {
        MLPRegressorOperation.HIDDEN_LAYER_SIZES_PARAM: '(100,100,9)'
    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = MLPRegressorOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = MLPRegressor(hidden_layer_sizes=(100,100,9),
        activation='relu', solver='adam', alpha=0.0001,
        max_iter=200, random_state=None, tol=0.0001)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_mlp_regressor_with_params_success():
    params = {
        MLPRegressorOperation.HIDDEN_LAYER_SIZES_PARAM: '(100,10,9)',
        MLPRegressorOperation.ACTIVATION_PARAM:
            MLPRegressorOperation.ACTIVATION_PARAM_LOG,
        MLPRegressorOperation.SEED_PARAM: 9,
        MLPRegressorOperation.SOLVER_PARAM:
            MLPRegressorOperation.SOLVER_PARAM_LBFGS,
        MLPRegressorOperation.MAX_ITER_PARAM: 1000,
        MLPRegressorOperation.ALPHA_PARAM: 0.01,
        MLPRegressorOperation.TOLERANCE_PARAM: 0.1,
    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = MLPRegressorOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = MLPRegressor(hidden_layer_sizes=(100,10,9),
        activation='logistic', solver='lbfgs', alpha=0.01,
        max_iter=1000, random_state=9, tol=0.1)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_mlp_regressor_wrong_value_param_failure():
    params = {
        MLPRegressorOperation.HIDDEN_LAYER_SIZES_PARAM: '100.100,'
    }
    n_out = {'algorithm': 'regressor_1'}
    with pytest.raises(ValueError):
        MLPRegressorOperation(params, named_inputs={}, named_outputs=n_out)


'''
    Random Forest Regressor Operation
'''


def test_randomforestregressor_minimum_params_success():
    params = {
    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = RandomForestRegressorOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = RandomForestRegressor(n_estimators=10,
            max_features='auto',
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=None)""")

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_randomforestregressor_with_params_success():
    params = {

        RandomForestRegressorOperation.MAX_FEATURES_PARAM: 'sqrt',
        RandomForestRegressorOperation.MAX_DEPTH_PARAM: 10,
        RandomForestRegressorOperation.MIN_LEAF_PARAM: 3,
        RandomForestRegressorOperation.MIN_SPLIT_PARAM: 4,
        RandomForestRegressorOperation.N_ESTIMATORS_PARAM: 9,
        RandomForestRegressorOperation.SEED_PARAM: -9

    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = RandomForestRegressorOperation(
        params, named_inputs={}, named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = RandomForestRegressor(n_estimators=9,
            max_features='sqrt',
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=3,
            random_state=-9)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_randomforestregressor_wrong_value_param_failure():
    params = {
        RandomForestRegressorOperation.MAX_DEPTH_PARAM: -1
    }
    n_in = {}
    n_out = {'algorithm': 'regressor_1'}
    with pytest.raises(ValueError):
        RandomForestRegressorOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)


'''
    Gradient Boosting Regressor Operation
'''


def test_sgd_regressor_minimum_params_success():
    params = {
    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = SGDRegressorOperation(params, named_inputs={},
                                        named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = SGDRegressor(alpha=0.0001, l1_ratio=0.15, max_iter=1000,
                                   tol=0.001, random_state=None)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_sgd_regressor_with_params_success():
    params = {
        SGDRegressorOperation.ELASTIC_PARAM: 0.10,
        SGDRegressorOperation.ALPHA_PARAM: 0.11,
        SGDRegressorOperation.TOLERANCE_PARAM: 0.12,
        SGDRegressorOperation.MAX_ITER_PARAM: 13,
        SGDRegressorOperation.SEED_PARAM: 14,
    }
    n_out = {'algorithm': 'regressor_1'}

    instance_lr = SGDRegressorOperation(params, named_inputs={},
                                        named_outputs=n_out)

    code = instance_lr.generate_code()
    expected_code = dedent("""
        regressor_1 = SGDRegressor(alpha=0.11, l1_ratio=0.10, max_iter=13,
                                   tol=0.12, random_state=14)""")
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_sgd_regressor_wrong_value_param_failure():
    params = {
        SGDRegressorOperation.ALPHA_PARAM: -1.0
    }
    n_in = {}
    n_out = {'algorithm': 'regressor_1'}
    with pytest.raises(ValueError):
        SGDRegressorOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    Regression Operation
'''


def test_regressor_model_operation_missing_output_failure():
    params = {
        RegressionModelOperation.FEATURES_PARAM: 'f',
    }
    n_in = {'algorithm': 'r', 'train input data': 't'}
    n_out = {'output data': 'out'}

    with pytest.raises(ValueError):
        regressor = RegressionModelOperation(params, named_inputs=n_in,
                                             named_outputs=n_out)
        regressor.generate_code()


def test_regressor_operation_success():
    params = {
            RegressionModelOperation.FEATURES_PARAM: 'f',
            RegressionModelOperation.LABEL_PARAM: 'l'
    }
    n_in = {'algorithm': 'regressor', 'train input data': 'train_data'}
    n_out = {'output data': 'out_data'}

    instance = RegressionModelOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        algorithm = regressor
        out_data = train_data.copy()
        X_train = train_data['f'].values.tolist()
        if 'IsotonicRegression' in str(algorithm):
            X_train = np.ravel(X_train)
        y = train_data['l'].values.tolist()
        model_1 = algorithm.fit(X_train, y)
        out_data['prediction'] = algorithm.predict(X_train).tolist()
    """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_regressor_operation_with_model_success():
    params = {
            RegressionModelOperation.FEATURES_PARAM: 'f',
            RegressionModelOperation.LABEL_PARAM: 'l'
    }
    n_in = {'algorithm': 'regressor', 'train input data': 'train_data'}
    n_out = {'model': 'model_data'}

    instance = RegressionModelOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        algorithm = regressor
        out_task_1 = train_data.copy()
        X_train = train_data['f'].values.tolist()
        if 'IsotonicRegression' in str(algorithm):
            X_train = np.ravel(X_train)
        y = train_data['l'].values.tolist()
        model_data = algorithm.fit(X_train, y)
        out_task_1['prediction'] = algorithm.predict(X_train).tolist()
    """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)