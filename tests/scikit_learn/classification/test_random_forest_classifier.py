from tests.scikit_learn import util
from tests.scikit_learn.util import get_label_data, get_X_train_data
from juicer.scikit_learn.classification_operation import \
    RandomForestClassifierModelOperation, RandomForestClassifierOperation, \
    ClassificationModelOperation
from sklearn.ensemble import RandomForestClassifier
import pytest
import pandas as pd
import numpy as np


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# RandomForestClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_random_forest_classifier_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_prediction_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'prediction': 'success'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[2] == 'success'


def test_random_forest_classifier_n_estimators_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'n_estimators': 5,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=5,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_max_depth_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'max_depth': 2,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=2, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_min_samples_split_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'min_samples_split': 4,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=4,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_min_samples_leaf_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'min_samples_leaf': 2,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=2, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_seed_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'seed': 2002},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=2002,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_criterion_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'criterion': 'entropy',
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='entropy',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_min_weight_fraction_leaf_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'min_weight_fraction_leaf': 0.3,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.3,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_max_features_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'max_features': 1,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=1, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_max_leaf_nodes_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'max_leaf_nodes': 2,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=2,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_min_impurity_decrease_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'min_impurity_decrease': 0.5,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.5, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_bootstrap_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'bootstrap': 0,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=False,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_oob_score_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'oob_score': 1,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=True, n_jobs=None, ccp_alpha=0.0,
                                     max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_n_jobs_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'n_jobs': 2,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=2, ccp_alpha=0.0,
                                     max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_ccp_alpha_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'ccp_alpha': 0.5,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.5, max_samples=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_max_samples_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'max_samples': 1,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = RandomForestClassifier(n_estimators=10,
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, random_state=1,
                                     criterion='gini',
                                     min_weight_fraction_leaf=0.0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, bootstrap=True,
                                     oob_score=False, n_jobs=None,
                                     ccp_alpha=0.0, max_samples=1 / 100.0)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_random_forest_classifier_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


def test_random_forest_classifier_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = RandomForestClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_random_forest_classifier_missing_label_param_fail():
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
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        RandomForestClassifierModelOperation(**arguments)
    assert "Parameters 'features' and 'label' must be informed for task" \
           " ClassificationModelOperation" in str(val_err.value)


def test_random_forest_classifier_missing_features_param_fail():
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
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        RandomForestClassifierModelOperation(**arguments)
    assert "Parameters 'features' must be informed for task" \
           " RandomForestClassifierOperation" in str(val_err.value)


def test_random_forest_classifier_invalid_min_weight_fraction_leaf_param_fail():
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'min_weight_fraction_leaf': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        RandomForestClassifierModelOperation(**arguments)
    assert f"Parameter 'min_weight_fraction_leaf' must be x>=0.0 and x<=0.5 for" \
           f" task {RandomForestClassifierOperation}" in str(val_err.value)


def test_random_forest_classifier_invalid_max_samples_param_fail():
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'max_samples': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        RandomForestClassifierModelOperation(**arguments)
    assert "Parameter 'max_samples' must be x>0 and x<100, or empty, " \
           f"case you want to use a fully sample for task" \
           f" {RandomForestClassifierOperation}" in str(val_err.value)


def test_random_forest_classifier_invalid_max_features_param_fail():
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'max_features': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        RandomForestClassifierModelOperation(**arguments)
    assert f"Parameter 'max_features' must be x>=0  for task" \
           f" {RandomForestClassifierOperation}" in str(val_err.value)


def test_random_forest_classifier_invalid_ccp_min_impurity_params_fail():
    vals = [
        'ccp_alpha',
        'min_impurity_decrease'
    ]
    for par in vals:
        arguments = {
            'parameters': {'label': ['sepalwidth'],
                           'features': ['sepallength', 'sepalwidth'],
                           'multiplicity': {'train input data': 0},
                           par: -1},
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        arguments = util.add_minimum_ml_args(arguments)
        with pytest.raises(ValueError) as val_err:
            RandomForestClassifierModelOperation(**arguments)
        assert f"Parameter '{par}' must be x>=0 for task" \
               f" {RandomForestClassifierOperation}" in str(val_err.value)


def test_random_forest_classifier_invalid_min_leaf_n_estimators_params_fail():
    vals = [
        'min_samples_split',
        'min_samples_leaf',
        'n_estimators'
    ]
    for par in vals:
        arguments = {
            'parameters': {'label': ['sepalwidth'],
                           'features': ['sepallength', 'sepalwidth'],
                           'multiplicity': {'train input data': 0},
                           par: -1},
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        arguments = util.add_minimum_ml_args(arguments)
        with pytest.raises(ValueError) as val_err:
            RandomForestClassifierModelOperation(**arguments)
        assert f"Parameter '{par}' must be x>0 for task" \
               f" {RandomForestClassifierOperation}" in str(val_err.value)
