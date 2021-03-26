from tests.scikit_learn import util
from juicer.scikit_learn.classification_operation import \
    LogisticRegressionOperation, LogisticRegressionModelOperation
from sklearn.linear_model import LogisticRegression
from tests.scikit_learn.util import get_label_data, get_X_train_data
import pandas as pd
import numpy as np
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# LogisticRegression
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_logistic_regression_solver_liblinear_and_penalty_params_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'penalty': 'l1',
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = LogisticRegression(tol=0.0001, C=1.0,
                                 max_iter=100, solver='liblinear',
                                 random_state=1,
                                 penalty='l1', dual=False,
                                 fit_intercept=True,
                                 intercept_scaling=1.0,
                                 multi_class='ovr', n_jobs=None,
                                 l1_ratio=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_logistic_regression_solver_newton_gc_and_tol_params_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'solver': 'newton-cg',
                       'tol': 0.1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = LogisticRegression(tol=0.1, C=1.0, max_iter=100,
                                 solver='newton-cg', random_state=None,
                                 penalty='l2',
                                 dual=False, fit_intercept=True,
                                 intercept_scaling=1.0, multi_class='ovr',
                                 n_jobs=None, l1_ratio=None)
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_logistic_regression_solver_lbfgs_and_regularization_params_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'solver': 'lbfgs',
                       'regularization': 2.0},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = LogisticRegression(tol=0.0001, C=2.0, max_iter=100,
                                 solver='lbfgs', random_state=None,
                                 penalty='l2',
                                 dual=False, fit_intercept=True,
                                 intercept_scaling=1.0, multi_class='ovr',
                                 n_jobs=None, l1_ratio=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_logistic_regression_solver_sag_and_max_iter_params_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'solver': 'sag',
                       'max_iter': 20,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = LogisticRegression(tol=0.0001, C=1.0,
                                 max_iter=20, solver='sag', random_state=1,
                                 penalty='l2', dual=False,
                                 fit_intercept=True,
                                 intercept_scaling=1.0,
                                 multi_class='ovr', n_jobs=None,
                                 l1_ratio=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_logistic_regression_solver_saga_and_random_state_params_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'solver': 'saga',
                       'random_state': 2002},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = LogisticRegression(tol=0.0001, C=1.0, max_iter=100,
                                 solver='saga', random_state=2002,
                                 penalty='l2',
                                 dual=False, fit_intercept=True,
                                 intercept_scaling=1.0, multi_class='ovr',
                                 n_jobs=None, l1_ratio=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_logistic_regression_dual_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'dual': True,
                       'penalty': 'l2',
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = LogisticRegression(tol=0.0001, C=1.0, max_iter=100,
                                 solver='liblinear', random_state=1,
                                 penalty='l2',
                                 dual=True, fit_intercept=True,
                                 intercept_scaling=1.0, multi_class='ovr',
                                 n_jobs=None, l1_ratio=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_logistic_regression_fit_intercept_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'fit_intercept': False},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = LogisticRegression(tol=0.0001, C=1.0, max_iter=100,
                                 solver='liblinear', random_state=None,
                                 penalty='l2',
                                 dual=False, fit_intercept=False,
                                 intercept_scaling=1.0, multi_class='ovr',
                                 n_jobs=None, l1_ratio=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_logistic_regression_intercept_scaling_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'intercept_scaling': 2.0},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = LogisticRegression(tol=0.0001, C=1.0, max_iter=100,
                                 solver='liblinear', random_state=None,
                                 penalty='l2',
                                 dual=False, fit_intercept=True,
                                 intercept_scaling=2.0, multi_class='ovr',
                                 n_jobs=None, l1_ratio=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_logistic_regression_multi_class_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'multi_class': 'auto'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = LogisticRegression(tol=0.0001, C=1.0, max_iter=100,
                                 solver='liblinear', random_state=None,
                                 penalty='l2',
                                 dual=False, fit_intercept=True,
                                 intercept_scaling=1.0, multi_class='auto',
                                 n_jobs=None, l1_ratio=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_logistic_regression_n_jobs_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'n_jobs': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = LogisticRegression(tol=0.0001, C=1.0, max_iter=100,
                                 solver='liblinear', random_state=None,
                                 penalty='l2',
                                 dual=False, fit_intercept=True,
                                 intercept_scaling=1.0, multi_class='ovr',
                                 n_jobs=2, l1_ratio=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_logistic_regression_l1_ratio_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'l1_ratio': 0.5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = LogisticRegression(tol=0.0001, C=1.0, max_iter=100,
                                 solver='liblinear', random_state=None,
                                 penalty='l2',
                                 dual=False, fit_intercept=True,
                                 intercept_scaling=1.0, multi_class='ovr',
                                 n_jobs=None, l1_ratio=0.5)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_logistic_regression_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    assert instance.generate_code() is None


def test_logistic_regression_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = LogisticRegressionModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_logistic_regression_missing_label_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"Parameters 'label' must be informed for task" \
           f" {LogisticRegressionOperation}" in str(val_err.value)


def test_logistic_regression_missing_features_param_fail():
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"Parameters 'features' must be informed for task" \
           f" {LogisticRegressionOperation}" in str(
        val_err.value)


def test_logistic_regression_invalid_tol_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'tol': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"Parameter 'tol' must be x>0 for task" \
           f" {LogisticRegressionOperation}" in str(val_err.value)


def test_lgs_reg_solver_liblinear_invalid_fit_intercept_intercept_scaling_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'solver': 'liblinear',
                       'fit_intercept': 1,
                       'intercept_scaling': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"Parameter 'intercept_scaling' must be x>0 for task" \
           f" {LogisticRegressionOperation}" in str(val_err.value)


def test_logistic_regression_invalid_n_jobs_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'n_jobs': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"Parameter 'n_jobs' must be x>0 for task " \
           f"{LogisticRegressionOperation}" in str(val_err.value)


def test_logistic_regression_invalid_regularization_max_iter_params_fail():
    vals = [
        'regularization',
        'max_iter'
    ]
    for par in vals:
        arguments = {
            'parameters': {'features': ['sepallength', 'sepalwidth'],
                           'label': ['sepalwidth'],
                           'multiplicity': {'train input data': 0},
                           par: -1},
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        util.add_minimum_ml_args(arguments)
        with pytest.raises(ValueError) as val_err:
            LogisticRegressionModelOperation(**arguments)
        assert f"Parameter '{par}' must be x>0 for task " \
               f"{LogisticRegressionOperation}" in str(val_err.value)


def test_logistic_regression_invalid_newton_cg_dual_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'solver': 'newton-cg',
                       'dual': True},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"For 'newton-cg' solver supports only dual=False for task " \
           f"{LogisticRegressionOperation}" in str(val_err.value)


def test_logistic_regression_invalid_liblinear_multinomial_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'solver': 'liblinear',
                       'multi_class': 'multinomial'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert 'Parameter "solver" does not support multi_class="multinomial"' in \
           str(val_err.value)


def test_logistic_regression_elastic_net_penalty_invalid_l1_ratio_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'solver': 'saga',
                       'l1_ratio': -1,
                       'penalty': 'elasticnet'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"Parameter 'l1_ratio' must be 0 <= x <= 1 for task" \
           f" {LogisticRegressionOperation}" in str(val_err.value)


def test_logistic_regression_elastic_net_penalty_invalid_l1_ratio_param_fail_2():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'solver': 'saga',
                       'l1_ratio': None,
                       'penalty': 'elasticnet'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        LogisticRegressionModelOperation(**arguments)
    assert f"Parameter 'l1_ratio' must be 0 <= x <= 1 for task" \
           f" {LogisticRegressionOperation}" in str(val_err.value)
