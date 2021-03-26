from tests.scikit_learn import util
from juicer.scikit_learn.classification_operation \
    import MLPClassifierModelOperation, MLPClassifierOperation
from sklearn.neural_network import MLPClassifier
from tests.scikit_learn.util import get_label_data, get_X_train_data
import pytest
import pandas as pd
import numpy as np


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# MLPClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_mlp_classifier_success():
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
                       'label': ['sepalwidth'],
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_hidden_layer_sizes_param_success():
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
                       'label': ['sepalwidth'],
                       'hidden_layer_sizes': '(1, 100)',
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(1, 100), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_activation_param_success():
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
                       'label': ['sepalwidth'],
                       'activation': 'tanh',
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='tanh',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_solver_param_success():
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
                       'label': ['sepalwidth'],
                       'solver': 'lbfgs',
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='lbfgs', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_alpha_param_success():
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
                       'label': ['sepalwidth'],
                       'alpha': 0.1,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.1, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_max_iter_param_success():
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
                       'label': ['sepalwidth'],
                       'max_iter': 100,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=100,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_tol_param_success():
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
                       'label': ['sepalwidth'],
                       'tol': 0.1,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.1, random_state=1, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_seed_param_success():
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
                       'label': ['sepalwidth'],
                       'seed': 2002},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=2002, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_batch_size_param_success():
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
                       'label': ['sepalwidth'],
                       'batch_size': 2,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size=2,
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_learning_rate_param_success():
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
                       'label': ['sepalwidth'],
                       'solver': 'sgd',
                       'learning_rate': 'adaptive',
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='sgd', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate='adaptive', nesterovs_momentum=True,
                            power_t=0.5, momentum=0.9, learning_rate_init=0.001,
                            shuffle=True, early_stopping=False,
                            n_iter_no_change=10)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_learning_rate_init_param_success():
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
                       'label': ['sepalwidth'],
                       'learning_rate_init': 0.1,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate_init=0.1, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_power_t_param_success():
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
                       'label': ['sepalwidth'],
                       'solver': 'sgd',
                       'power_t': 0.8,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='sgd', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate='constant', nesterovs_momentum=True,
                            power_t=0.8, momentum=0.9, learning_rate_init=0.001,
                            shuffle=True, early_stopping=False,
                            n_iter_no_change=10)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_shuffle_param_success():
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
                       'label': ['sepalwidth'],
                       'shuffle': 0,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate_init=0.001, shuffle=False,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_momentum_param_success():
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
                       'label': ['sepalwidth'],
                       'momentum': 0.5,
                       'solver': 'sgd',
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='sgd', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate='constant', nesterovs_momentum=True,
                            power_t=0.5, momentum=0.5, learning_rate_init=0.001,
                            shuffle=True, early_stopping=False,
                            n_iter_no_change=10)
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_nesterovs_momentum_param_success():
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
                       'label': ['sepalwidth'],
                       'solver': 'sgd',
                       'nesterovs_momentum': 0,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='sgd', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate='constant', nesterovs_momentum=False,
                            power_t=0.5, momentum=0.9, learning_rate_init=0.001,
                            shuffle=True, early_stopping=False,
                            n_iter_no_change=10)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_early_stopping_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    df.iloc[3:5, 1] = 7.0
    df.iloc[5:9, 1] = 2.0
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'early_stopping': 1,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=True, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_validation_fraction_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    df.iloc[3:5, 1] = 7.0
    df.iloc[5:9, 1] = 2.0
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'early_stopping': 1,
                       'validation_fraction': 0.3,
                       'solver': 'sgd',
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1),
                            activation='relu', solver='sgd', alpha=0.0001,
                            max_iter=200, tol=0.0001, random_state=1,
                            batch_size='auto', learning_rate='constant',
                            nesterovs_momentum=True, power_t=0.5,
                            momentum=0.9, learning_rate_init=0.001,
                            shuffle=True, early_stopping=True,
                            n_iter_no_change=10, validation_fraction=0.3)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_beta_1_beta_2_params_success():
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
                       'label': ['sepalwidth'],
                       'beta1': 0.5, 'beta2': 0.4, 'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.5, beta_2=0.4, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_epsilon_param_success():
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
                       'label': ['sepalwidth'],
                       'epsilon': 1e-05,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-05)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_n_iter_no_change_param_success():
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
                       'label': ['sepalwidth'],
                       'n_iter_no_change': 20,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=20,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_max_fun_param_success():
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
                       'label': ['sepalwidth'],
                       'solver': 'lbfgs',
                       'max_fun': 10000,
                       'seed': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='lbfgs', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=1, max_fun=10000)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_prediction_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'prediction': 'success'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[2] == 'success'


def test_mlp_classifier_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth']},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


def test_mlp_classifier_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth']},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_mlp_classifier_invalid_tol_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'tol': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter 'tol' must be x > 0 for task " \
           f"{MLPClassifierOperation}" in str(val_err)


def test_mlp_classifier_invalid_max_iter_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'max_iter': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter 'max_iter' must be x > 0 for task " \
           f"{MLPClassifierOperation}" in str(val_err.value)


def test_mlp_classifier_invalid_alpha_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'alpha': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter 'alpha' must be x >= 0 for task " \
           f"{MLPClassifierOperation}" in str(val_err.value)


def test_mlp_classifier_invalid_hidden_layers_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'hidden_layer_sizes': '(1)'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter 'hidden_layer_sizes' must be a tuple with the size of" \
           f" each layer for task {MLPClassifierOperation}" in str(val_err.value)


def test_mlp_classifier_invalid_momentum_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'solver': 'sgd',
                       'momentum': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter 'momentum' must be x between 0 and 1 for task " \
           f"{MLPClassifierOperation}" in str(val_err.value)


def test_mlp_classifier_invalid_learning_rate_init_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'solver': 'sgd',
                       'learning_rate_init': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter 'learning_rate_init' must be x > 0 for task" \
           f" {MLPClassifierOperation}" in \
           str(val_err.value)


def test_mlp_classifier_invalid_n_iter_no_change_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'solver': 'sgd',
                       'n_iter_no_change': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter 'n_iter_no_change' must be x > 0 for task" \
           f" {MLPClassifierOperation}" in str(val_err.value)


def test_mlp_classifier_invalid_validation_fraction_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'early_stopping': 1,
                       'validation_fraction': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert "Parameter 'validation_fraction' must be x between 0 and 1" \
           in str(val_err.value)


def test_mlp_classifier_invalid_beta_1_beta_2_params_fail():
    vals = [
        'beta1',
        'beta2'
    ]
    for par in vals:
        arguments = {
            'parameters': {'features': ['sepallength', 'sepalwidth'],
                           'multiplicity': {'train input data': 0},
                           'label': ['sepalwidth'],
                           'solver': 'adam',
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
            MLPClassifierModelOperation(**arguments)
        assert f"Parameter '{par}' must be in [0, 1) for task" \
               f" {MLPClassifierOperation}" in str(val_err.value)


def test_mlp_classifier_invalid_max_fun_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'solver': 'lbfgs',
                       'max_fun': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        MLPClassifierModelOperation(**arguments)
    assert f"Parameter 'max_fun' must be x > 0 for task" \
           f" {MLPClassifierOperation}" in str(val_err.value)
