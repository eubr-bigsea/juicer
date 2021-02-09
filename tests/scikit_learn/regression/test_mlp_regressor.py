from tests.scikit_learn import util
from juicer.scikit_learn.regression_operation import MLPRegressorOperation
from tests.scikit_learn.util import get_X_train_data, get_label_data
from sklearn.neural_network import MLPRegressor
import pytest
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# MLPRegressor
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_mlp_regressor_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
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
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto')
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_layer_sizes_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'layer_sizes': '(10, 200, 10)'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(10, 200, 10), activation='relu',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto')
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_identity_activation_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'activation': 'identity'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='identity',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto')
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_logistic_activation_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'activation': 'logistic'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='logistic',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto')
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_tanh_activation_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'activation': 'tanh'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='tanh',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto')
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_lbfgs_solver_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'solver': 'lbfgs'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='lbfgs', alpha=0.0001, max_iter=200,
                           tol=0.0001, batch_size='auto')
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_sgd_solver_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'solver': 'sgd'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='sgd', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto')
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_alpha_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'alpha': 0.01},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='adam', alpha=0.01, max_iter=200, tol=0.0001,
                           batch_size='auto')
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_max_iter_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_iter': 10},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='adam', alpha=0.0001, max_iter=10, tol=0.0001,
                           batch_size='auto')
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_tol_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'tol': 0.01},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.01,
                           batch_size='auto')
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_random_state_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'random_state': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.0001,
                           random_state=2, batch_size='auto')
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_mlp_regressor_batch_size_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'batch_size': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size=2)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_learning_rate_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'solver': 'sgd',
                       'learning_rate': 'adaptive'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='sgd', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto', learning_rate='adaptive',
                           power_t='0.5', momentum=0.9, nesterovs_momentum=True,
                           learning_rate_init=0.001, shuffle=True,
                           n_iter_no_change=10, early_stopping=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_prediction_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
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
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[4] == 'success'


def test_mlp_regressor_learning_rate_init_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'solver': 'adam',
                       'learning_rate_init': 0.2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto', beta_1=0.9, beta_2=0.999,
                           epsilon=1e-08, learning_rate_init=0.2, shuffle=True,
                           n_iter_no_change=10, early_stopping=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_power_t_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'solver': 'sgd',
                       'power_t': 0.2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='sgd', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto', learning_rate='constant',
                           power_t=0.2, momentum=0.9, nesterovs_momentum=True,
                           learning_rate_init=0.001, shuffle=True,
                           n_iter_no_change=10, early_stopping=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 2.0)


def test_mlp_regressor_shuffle_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'shuffle': 0,
                       'solver': 'adam'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor()
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_n_iter_no_change_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'n_iter_no_change': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto', beta_1=0.9, beta_2=0.999,
                           epsilon=1e-08, learning_rate_init=0.001, shuffle=True,
                           n_iter_no_change=1, early_stopping=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_momentum_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'solver': 'sgd',
                       'momentum': 0.1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1),
                                     activation='relu', solver='sgd',
                                     alpha=0.0001, max_iter=200, tol=0.0001,
                                     batch_size='auto', learning_rate='constant',
                                     power_t=0.5, momentum=0.1,
                                     nesterovs_momentum=True,
                                     learning_rate_init=0.001, shuffle=True,
                                     n_iter_no_change=10, early_stopping=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_nesterovs_momentum_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'nesterovs_momentum': 0,
                       'solver': 'sgd'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='sgd', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto', learning_rate='constant',
                           power_t=0.5, momentum=0.9, nesterovs_momentum=False,
                           learning_rate_init=0.001, shuffle=True,
                           n_iter_no_change=10, early_stopping=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_early_stopping_and_validation_fraction_params_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'early_stopping': 1,
                       'validation_fraction': 0.2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto', validation_fraction=0.2,
                           beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                           learning_rate_init=0.001, shuffle=True,
                           n_iter_no_change=10, early_stopping=True)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_beta_1_beta_2_params_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'beta_1': 0.5,
                       'beta_2': 0.6},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto', beta_1=0.5, beta_2=0.6,
                           epsilon=1e-08, learning_rate_init=0.001, shuffle=True,
                           n_iter_no_change=10, early_stopping=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_epsilon_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'epsilon': 0.1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = MLPRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = MLPRegressor(hidden_layer_sizes=(1, 100, 1), activation='relu',
                           solver='adam', alpha=0.0001, max_iter=200, tol=0.0001,
                           batch_size='auto', beta_1=0.9, beta_2=0.999,
                           epsilon=0.1, learning_rate_init=0.001, shuffle=True,
                           n_iter_no_change=10, early_stopping=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 10e-05, 10e+05)


def test_mlp_regressor_no_output_implies_no_code_success():
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
    instance = MLPRegressorOperation(**arguments)
    assert instance.generate_code() is None


def test_mlp_regressor_missing_input_implies_no_code_success():
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
    instance = MLPRegressorOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_mlp_regressor_invalid_layer_sizes_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'layer_sizes': (1, 2)},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        MLPRegressorOperation(**arguments)
    assert "Parameter 'layer_sizes' must be a tuple with the size of each" \
           " layer for task" in str(val_err)


def test_mlp_regressor_multiple_invalid_params_fail():
    pars = [
        'alpha',
        'max_iter',
        'tol'
    ]
    for val in pars:
        arguments = {
            'parameters': {f'features': ['sepallength', 'sepalwidth'],
                           'multiplicity': {'train input data': 0},
                           'label': ['sepallength'],
                           val: -1},
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        with pytest.raises(ValueError) as val_err:
            MLPRegressorOperation(**arguments)
        assert f"Parameter '{val}' must be x>=0 for task" in str(val_err.value)


def test_mlp_regressor_invalid_momentum_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'momentum': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        MLPRegressorOperation(**arguments)
    assert "Parameter 'momentum' must be x between" \
           " 0 and 1 for task" in str(val_err.value)


def test_mlp_regressor_invalid_beta_1_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'beta_1': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        MLPRegressorOperation(**arguments)
    assert "Parameter 'beta_1' must be in [0, 1) for task" in str(val_err.value)


def test_mlp_regressor_invalid_beta_2_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'beta_2': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        MLPRegressorOperation(**arguments)
    assert "Parameter 'beta_2' must be in [0, 1) for task" in str(val_err.value)
