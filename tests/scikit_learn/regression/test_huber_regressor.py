from tests.scikit_learn import util
from sklearn.linear_model import HuberRegressor
from juicer.scikit_learn.regression_operation import HuberRegressorOperation
import pytest
import pandas as pd

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# HuberRegressor
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_huber_regressor_success():
    df = util.iris(['sepallength', 'sepalwidth'],
                   size=10)
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
    instance = HuberRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    test_out = test_df
    X_train = util.get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = util.get_label_data(test_df, ['sepallength'])

    model_1 = HuberRegressor(epsilon=1.35, max_iter=100,
                             alpha=0.0001, tol=1e-05, fit_intercept=True,
                             warm_start=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_huber_regressor_epsilon_param_success():
    df = util.iris(['sepallength', 'sepalwidth'],
                   size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'epsilon': 1.6},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = HuberRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    test_out = test_df
    X_train = util.get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = util.get_label_data(test_df, ['sepallength'])

    model_1 = HuberRegressor(epsilon=1.6, max_iter=100,
                             alpha=0.0001, tol=1e-05, fit_intercept=True,
                             warm_start=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_huber_regressor_max_iter_param_success():
    df = util.iris(['sepallength', 'sepalwidth'],
                   size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_iter': 5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = HuberRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    test_out = test_df
    X_train = util.get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = util.get_label_data(test_df, ['sepallength'])

    model_1 = HuberRegressor(epsilon=1.35, max_iter=5,
                             alpha=0.0001, tol=1e-05, fit_intercept=True,
                             warm_start=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_huber_regressor_alpha_param_success():
    df = util.iris(['sepallength', 'sepalwidth'],
                   size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'alpha': 0.2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = HuberRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    test_out = test_df
    X_train = util.get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = util.get_label_data(test_df, ['sepallength'])

    model_1 = HuberRegressor(epsilon=1.35, max_iter=100,
                             alpha=0.2, tol=1e-05, fit_intercept=True,
                             warm_start=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_huber_regressor_tol_param_success():
    df = util.iris(['sepallength', 'sepalwidth'],
                   size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'tol': 0.5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = HuberRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_df
    X_train = util.get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = util.get_label_data(test_df, ['sepallength'])

    model_1 = HuberRegressor(epsilon=1.35, max_iter=100,
                             alpha=0.0001, tol=0.5, fit_intercept=True,
                             warm_start=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_huber_regressor_fit_intercept_param_success():
    df = util.iris(['sepallength', 'sepalwidth'],
                   size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'fit_intercept': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = HuberRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    test_out = test_df
    X_train = util.get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = util.get_label_data(test_df, ['sepallength'])

    model_1 = HuberRegressor(epsilon=1.35, max_iter=100,
                             alpha=0.0001, tol=1e-05, fit_intercept=False,
                             warm_start=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_huber_regressor_prediction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'],
                   size=10)
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
    instance = HuberRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[2] == 'success'


def test_huber_regressor_no_output_implies_no_code_success():
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
    instance = HuberRegressorOperation(**arguments)
    assert instance.generate_code() is None


def test_huber_regressor_missing_input_implies_no_code_success():
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
    instance = HuberRegressorOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_huber_regressor_invalid_max_iter_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_iter': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        HuberRegressorOperation(**arguments)
    assert "Parameter 'max_iter' must be x>0 for task" in str(val_err.value)


def test_huber_regressor_invalid_alpha_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'alpha': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        HuberRegressorOperation(**arguments)
    assert "Parameter 'alpha' must be x>0 for task" in str(val_err.value)


def test_huber_regressor_invalid_epsilon_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'epsilon': 0.5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        HuberRegressorOperation(**arguments)
    assert "Parameter 'epsilon' must be x>1.0 for task" in str(val_err.value)


def test_huber_regressor_missing_multiplicity_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'],
                   size=10)
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepallength']},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = HuberRegressorOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert 'multiplicity' in str(key_err.value)


def test_huber_regressor_missing_features_param_fail():
    arguments = {
        'parameters': {'multiplicity': {'train input data': 0},
                       'label': ['sepallength']},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    with pytest.raises(KeyError) as key_err:
        HuberRegressorOperation(**arguments)
    assert "features" in str(key_err.value)


def test_huber_regressor_missing_label_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'],
                   size=10)
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
    instance = HuberRegressorOperation(**arguments)
    with pytest.raises(TypeError) as typ_err:
        util.execute(util.get_complete_code(instance),
                     {'df': df})
    assert "object of type 'NoneType' has no len()" in str(typ_err.value)
