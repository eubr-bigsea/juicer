from tests.scikit_learn import util
from juicer.scikit_learn.classification_operation import GBTClassifierOperation, \
    GBTClassifierModelOperation
from sklearn.ensemble import GradientBoostingClassifier
from tests.scikit_learn.util import get_label_data, get_X_train_data
import pytest
import pandas as pd
import numpy as np


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# GBTClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_gbt_classifier_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_learning_rate_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'learning_rate': 0.3,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.3,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_n_estimators_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'n_estimators': 50,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=50, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_max_depth_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_depth': 5,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=5, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_min_samples_split_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_samples_split': 5,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=5,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_min_samples_leaf_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_samples_leaf': 3,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=3,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_loss_param_success():
    df = pd.DataFrame(
        [[5, 3], [4, 3],
         [4, 3], [4, 3],
         [5, 3], [5, 3],
         [4, 3], [5, 3],
         [4, 2], [4, 3]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'loss': 'exponential',
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='exponential',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_random_state_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'random_state': 2002},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=2002, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_subsample_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'subsample': 0.5,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=0.5,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_criterion_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'criterion': 'mae',
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='mae',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_min_weight_fraction_leaf_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_weight_fraction_leaf': 0.5,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.5,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_min_impurity_decrease_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_impurity_decrease': 0.2,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.2, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_init_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'init': '"zero"',
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init='zero',
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_max_features_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_features': 'auto',
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features='auto',
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_max_leaf_nodes_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_leaf_nodes': 2,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=2, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_validation_fraction_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'validation_fraction': 0.2,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.2,
                                         n_iter_no_change=None, tol=0.0001)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_n_iter_no_change_param_success():
    df = util.iris(['sepallength', 'sepalwidth', ], size=42)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'n_iter_no_change': 4,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=4, tol=0.0001)

    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_tol_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth', ],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'tol': 0.1,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=1, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.1)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_gbt_classifier_prediction_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'prediction': 'success'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[2] == 'success'


def test_gbt_classifier_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength']},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


def test_gbt_classifier_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength']},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = GBTClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_gbt_classifier_multiple_invalid_params_fail():
    pars = [
        'min_samples_split',
        'min_samples_leaf',
        'learning_rate',
        'n_estimators',
        'max_depth'
    ]
    for arg in pars:
        arguments = {
            'parameters': {'features': ['sepallength', 'sepalwidth'],
                           'multiplicity': {'train input data': 0},
                           'label': ['sepallength'],
                           arg: -1},
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        util.add_minimum_ml_args(arguments)
        with pytest.raises(ValueError) as val_err:
            GBTClassifierModelOperation(**arguments)
        assert f"Parameter '{arg}' must be x>0 for task" in str(val_err.value)


def test_gbt_classifier_invalid_max_leafs_nodes_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_leaf_nodes': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        GBTClassifierModelOperation(**arguments)
    assert "Parameter 'max_leaf_nodes' must be None or x > 1 for task" in str(
        val_err.value)


def test_gbt_classifier_invalid_n_iter_no_change_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'n_iter_no_change': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        GBTClassifierModelOperation(**arguments)
    assert "Parameter 'n_iter_no_change' must be None or x > 0 for task" in str(
        val_err.value)


def test_gbt_classifier_invalid_validation_fraction_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'validation_fraction': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        GBTClassifierModelOperation(**arguments)
    assert "Parameter 'validation_fraction' must be 0 <= x =< 1 for task" in \
           str(val_err.value)


def test_gbt_classifier_invalid_subsample_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'subsample': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        GBTClassifierModelOperation(**arguments)
    assert "Parameter 'subsample' must be 0 < x =< 1 for task" in \
           str(val_err.value)


def test_gbt_classifier_invalid_min_weight_fraction_leaf_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_weight_fraction_leaf': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        GBTClassifierModelOperation(**arguments)
    assert "Parameter 'min_weight_fraction_leaf' must be 0.0" \
           " <= x =< 0.5 for task" in str(val_err.value)
