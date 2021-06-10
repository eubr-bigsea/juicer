from tests.scikit_learn import util
from juicer.scikit_learn.classification_operation import KNNClassifierOperation, \
    KNNClassifierModelOperation
from sklearn.neighbors import KNeighborsClassifier
from tests.scikit_learn.util import get_X_train_data, get_label_data
import pytest
import pandas as pd
import numpy as np


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# KNNClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_knn_classifier_success():
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
                       'label': ['sepallength']},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = KNNClassifierModelOperation(**arguments)
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params=None, n_jobs=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_knn_classifier_n_neighbors_param_success():
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
                       'n_neighbors': 10},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = KNNClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = KNeighborsClassifier(n_neighbors=10,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params=None, n_jobs=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_knn_classifier_weights_param_success():
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
                       'weights': 'distance'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = KNNClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='distance', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params=None, n_jobs=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_knn_classifier_algorithm_param_success():
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
                       'algorithm': 'brute'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = KNNClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='brute',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params=None, n_jobs=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_knn_classifier_leaf_size_param_success():
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
                       'leaf_size': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = KNNClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=2, p=2, metric='minkowski',
                                   metric_params=None, n_jobs=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_knn_classifier_p_param_success():
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
                       'p': 4},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = KNNClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=4, metric='minkowski',
                                   metric_params=None, n_jobs=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_knn_classifier_metric_param_success():
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
                       'metric': 'braycurtis'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = KNNClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='braycurtis',
                                   metric_params=None, n_jobs=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_knn_classifier_metric_params_param_success():
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
                       'metric_params': {'1': '1'}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = KNNClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params={'1': '1'}, n_jobs=None)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_knn_classifier_n_jobs_param_success():
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
                       'n_jobs': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = KNNClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params=None, n_jobs=2)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_knn_classifier_prediction_param_success():
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
    instance = KNNClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[2] == 'success'


def test_knn_classifier_no_output_implies_no_code_success():
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
    instance = KNNClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


def test_knn_classifier_missing_input_implies_no_code_success():
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
    instance = KNNClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_knn_classifier_invalid_invalid_n_neighbors_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'n_neighbors': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        KNNClassifierOperation(**arguments)
    assert f"Parameter 'n_neighbors' must be x>0 for task" \
           f" {KNNClassifierOperation}" in str(val_err.value)


def test_knn_classifier_invalid_p_param_fail():
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
                       'p': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        KNNClassifierOperation(**arguments)
    assert f"Parameter 'p' must be x>1 when parameter 'metric_params'" \
           f" is 'minkowski' for task {KNNClassifierOperation}" in str(
        val_err.value)
