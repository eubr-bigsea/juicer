from tests.scikit_learn import util
from juicer.scikit_learn.regression_operation import SGDRegressorOperation
import pytest
import pandas as pd
from sklearn.linear_model import SGDRegressor
from tests.scikit_learn.util import get_X_train_data, get_label_data

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# SGDRegressor:
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_sgd_regressor_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=0,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.01)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_sgd_regressor_alpha_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'alpha': 0.1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=0,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.1,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.01)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.01)


def test_sgd_regressor_l1_ratio_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'l1_ratio': 0.3},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=0,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.3,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.01)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_sgd_regressor_max_iter_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'max_iter': 500},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=0,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=500,
                           random_state=None,
                           eta0=0.1)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_sgd_regressor_tol_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'tol': 0.1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=0,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.01,
                           tol=0.1)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_sgd_regressor_random_state_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'random_state': 2002},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=0,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=2002,
                           eta0=0.01)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_sgd_regressor_prediction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'prediction': 'success'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[2] == 'success'


def test_sgd_regressor_power_t_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'power_t': 0.1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.1,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=0,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.01)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.01)


def test_sgd_regressor_early_stopping_and_validation_fraction_params_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'early_stopping': 1,
                       'validation_fraction': 0.3},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=True,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=0,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.01,
                           validation_fraction=0.3)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_sgd_regressor_loss_and_epsilon_params_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'loss': 'epsilon_insensitive',
                       'epsilon': 0.2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='epsilon_insensitive',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=0,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.01,
                           epsilon=0.2)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_sgd_regressor_n_iter_no_change_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'n_iter_no_change': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=2,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=0,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.01)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_sgd_regressor_penalty_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'penalty': 'l1'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l1',
                           fit_intercept=1,
                           verbose=0,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.01)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.01)


def test_sgd_regressor_fit_intercept_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'fit_intercept': 3},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=3,
                           verbose=0,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.01)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.01)


def test_sgd_regressor_eta0_and_learning_rate_params_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'eta0': 0.1,
                       'learning_rate': 'constant'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=0,
                           average=1,
                           learning_rate='constant',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.1)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 3.0e+11, 3.0e-11)


def test_sgd_regressor_verbose_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'verbose': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=1,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.01)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.01)


def test_sgd_regressor_average_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'average': 0},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=0,
                           average=0,
                           learning_rate='invscaling',
                           shuffle=True,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.01)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.01)


def test_sgd_regressor_shuffle_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'shuffle': 0},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = SGDRegressor(loss='squared_loss',
                           power_t=0.5,
                           early_stopping=False,
                           n_iter_no_change=5,
                           penalty='l2',
                           fit_intercept=1,
                           verbose=0,
                           average=1,
                           learning_rate='invscaling',
                           shuffle=False,
                           alpha=0.0001,
                           l1_ratio=0.15,
                           max_iter=1000,
                           random_state=None,
                           eta0=0.01)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 1)


def test_sgd_regressor_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = SGDRegressorOperation(**arguments)
    assert instance.generate_code() is None


def test_sgd_regressor_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = SGDRegressorOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_sgd_regressor_multiple_invalid_params_fail():
    params = ['alpha', 'max_iter',
              'n_iter_no_change', 'eta0']

    for val in params:
        arguments = {
            'parameters': {'features': ['sepallength', 'sepalwidth'],
                           'label': ['sepallength'],
                           'multiplicity': {'train input data': 0},
                           val: -1},
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        with pytest.raises(ValueError) as val_err:
            SGDRegressorOperation(**arguments)
        assert f"Parameter '{val}' must be x>0 for task" in str(val_err.value)


def test_sgd_regressor_invalid_random_state_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'random_state': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        SGDRegressorOperation(**arguments)
    assert "Parameter 'random_state' must be x >= 0 for task" in str(
        val_err.value)


def test_sgd_regressor_invalid_l1_ratio_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'l1_ratio': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        SGDRegressorOperation(**arguments)
    assert "Parameter 'l1_ratio' must be 0 <= x =< 1 for task" in str(
        val_err.value)


def test_sgd_regressor_invalid_validation_fraction_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'validation_fraction': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        SGDRegressorOperation(**arguments)
    assert "Parameter 'validation_fraction' must be 0 <= x =< 1 for task" in \
           str(val_err.value)
