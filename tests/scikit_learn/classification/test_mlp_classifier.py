from tests.scikit_learn import util
from juicer.scikit_learn.classification_operation \
    import MLPClassifierModelOperation
from sklearn.neural_network import MLPClassifier
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# MLPClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_mlp_classifier_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth']},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_hidden_layer_sizes_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'hidden_layer_sizes': '(1, 100)'},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(1, 100), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_activation_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'activation': 'tanh'},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='tanh',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_solver_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'solver': 'lbfgs'},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='lbfgs', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_alpha_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'alpha': 0.1},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.1, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_max_iter_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'max_iter': 100},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=100,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_tol_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'tol': 0.1},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.1, random_state=None, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_seed_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=2002, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_batch_size_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'batch_size': 2},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size=2,
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_learning_rate_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'solver': 'sgd',
                       'learning_rate': 'adaptive'},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='sgd', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate='adaptive', nesterovs_momentum=True,
                            power_t=0.5, momentum=0.9, learning_rate_init=0.001,
                            shuffle=True, early_stopping=False,
                            n_iter_no_change=10)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_learning_rate_init_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'learning_rate_init': 0.1},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate_init=0.1, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_power_t_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'solver': 'sgd',
                       'power_t': 0.8},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='sgd', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate='constant', nesterovs_momentum=True,
                            power_t=0.8, momentum=0.9, learning_rate_init=0.001,
                            shuffle=True, early_stopping=False,
                            n_iter_no_change=10)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_shuffle_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'shuffle': 0},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate_init=0.001, shuffle=False,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_momentum_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'momentum': 0.5,
                       'solver': 'sgd'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='sgd', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate='constant', nesterovs_momentum=True,
                            power_t=0.5, momentum=0.5, learning_rate_init=0.001,
                            shuffle=True, early_stopping=False,
                            n_iter_no_change=10)
    arguments = util.add_minimum_ml_args(arguments)
    instance = MLPClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_nesterovs_momentum_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'solver': 'sgd',
                       'nesterovs_momentum': 0},
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
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='sgd', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate='constant', nesterovs_momentum=False,
                            power_t=0.5, momentum=0.9, learning_rate_init=0.001,
                            shuffle=True, early_stopping=False,
                            n_iter_no_change=10)

    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_early_stopping_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    df.iloc[3:5, 1] = 7.0
    df.iloc[5:9, 1] = 2.0
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'early_stopping': 1},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=True, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_validation_fraction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    df.iloc[3:5, 1] = 7.0
    df.iloc[5:9, 1] = 2.0
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'early_stopping': 1,
                       'validation_fraction': 0.3,
                       'solver': 'sgd'},
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
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1),
                            activation='relu', solver='sgd', alpha=0.0001,
                            max_iter=200, tol=0.0001, random_state=None,
                            batch_size='auto', learning_rate='constant',
                            nesterovs_momentum=True, power_t=0.5,
                            momentum=0.9, learning_rate_init=0.001,
                            shuffle=True, early_stopping=True,
                            n_iter_no_change=10, validation_fraction=0.3)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_beta_1_beta_2_params_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'beta1': 0.5, 'beta2': 0.4},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.5, beta_2=0.4, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_epsilon_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'epsilon': 1e-05},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=10,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-05)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_n_iter_no_change_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'n_iter_no_change': 20},
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

    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='adam', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, batch_size='auto',
                            learning_rate_init=0.001, shuffle=True,
                            early_stopping=False, n_iter_no_change=20,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_max_fun_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'solver': 'lbfgs',
                       'max_fun': 10000},
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
    model_1 = MLPClassifier(hidden_layer_sizes=(100, 1), activation='relu',
                            solver='lbfgs', alpha=0.0001, max_iter=200,
                            tol=0.0001, random_state=None, max_fun=10000)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_mlp_classifier_prediction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    assert "Parameter 'tol' must be x > 0 for task" in str(val_err)


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
    assert "Parameter 'max_iter' must be x > 0 for task" in str(val_err.value)


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
    assert "Parameter 'alpha' must be x >= 0 for task" in str(val_err.value)


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
    assert "Parameter 'hidden_layer_sizes' must be a tuple with the size of" \
           " each layer for task" in str(val_err.value)


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
    assert "Parameter 'momentum' must be x between 0 and 1 for task" in \
           str(val_err.value)


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
    assert "Parameter 'learning_rate_init' must be x > 0 for task" in \
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
    assert "Parameter 'n_iter_no_change' must be x > 0 for task" in \
           str(val_err.value)


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
        assert f"Parameter '{par}' must be in [0, 1) for task" in str(
            val_err.value)


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
    assert "Parameter 'max_fun' must be x > 0 for task" in str(val_err.value)
