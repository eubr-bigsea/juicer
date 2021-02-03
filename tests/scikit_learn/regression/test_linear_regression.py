from tests.scikit_learn import util
from juicer.scikit_learn.regression_operation import LinearRegressionOperation
from sklearn.linear_model import ElasticNet
from tests.scikit_learn.util import get_X_train_data, get_label_data
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# LinearRegression:
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_linear_regression_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'normalize': False},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = ElasticNet(alpha=1.0, l1_ratio=0.5, tol=0.0001,
                         max_iter=1000, random_state=None,
                         normalize=False, positive=False,
                         fit_intercept=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_linear_regression_alpha_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'alpha': 0.5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = ElasticNet(alpha=0.5, l1_ratio=0.5, tol=0.0001,
                         max_iter=1000, random_state=None,
                         normalize=True, positive=False,
                         fit_intercept=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_linear_regression_l1_ratio_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'l1_ratio': 1.0},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = ElasticNet(alpha=1.0, l1_ratio=1.0, tol=0.0001,
                         max_iter=1000, random_state=None,
                         normalize=True, positive=False,
                         fit_intercept=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_linear_regression_normalize_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'normalize': True},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = ElasticNet(alpha=1.0, l1_ratio=0.5, tol=0.0001,
                         max_iter=1000, random_state=None,
                         normalize=True, positive=False,
                         fit_intercept=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_linear_regression_max_iter_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_iter': 500},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = ElasticNet(alpha=1.0, l1_ratio=0.5, tol=0.0001,
                         max_iter=500, random_state=None,
                         normalize=True, positive=False,
                         fit_intercept=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_linear_regression_tol_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'tol': 0.652},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = ElasticNet(alpha=1.0, l1_ratio=0.5, tol=0.652,
                         max_iter=1000, random_state=None,
                         normalize=False, positive=False,
                         fit_intercept=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_linear_regression_random_state_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
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
    instance = LinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = ElasticNet(alpha=1.0, l1_ratio=0.5, tol=0.0001,
                         max_iter=1000, random_state=2002,
                         normalize=False, positive=False,
                         fit_intercept=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_linear_regression_prediction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
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
    instance = LinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[2] == 'success'


def test_linear_regression_positive_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'positive': True},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = ElasticNet(alpha=1.0, l1_ratio=0.5, tol=0.0001,
                         max_iter=1000, random_state=None,
                         normalize=False, positive=True,
                         fit_intercept=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_linear_regression_fit_intercept_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'fit_intercept': True},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = ElasticNet(alpha=1.0, l1_ratio=0.5, tol=0.0001,
                         max_iter=1000, random_state=None,
                         normalize=False, positive=False,
                         fit_intercept=True)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_linear_regression_no_output_implies_no_code_success():
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
    instance = LinearRegressionOperation(**arguments)
    assert instance.generate_code() is None


def test_linear_regression_missing_input_implies_no_code_success():
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
    instance = LinearRegressionOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Success # # # # # # # # # #
def test_linear_regression_invalid_alpha_param_fail():
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
        LinearRegressionOperation(**arguments)
    assert "Parameter 'alpha' must be x>0 for task" in str(val_err.value)


def test_linear_regression_invalid_max_iter_param_fail():
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
        LinearRegressionOperation(**arguments)
    assert "Parameter 'max_iter' must be x>0 for task" in str(val_err.value)


def test_linear_regression_invalid_l1_ratio_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'l1_ratio': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        LinearRegressionOperation(**arguments)
    assert "Parameter 'l1_ratio' must be 0<=x<=1 for task" in str(val_err.value)


def test_linear_regression_invalid_random_state_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'random_state': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        LinearRegressionOperation(**arguments)
    assert "Parameter 'random_state' must be x>=0 for task" in str(val_err.value)


def test_linear_regression_missing_features_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
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
        LinearRegressionOperation(**arguments)
    assert "features" in str(key_err.value)


def test_linear_regression_missing_label_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
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
    instance = LinearRegressionOperation(**arguments)
    with pytest.raises(TypeError) as typ_err:
        util.execute(util.get_complete_code(instance),
                     {'df': df})
    assert "object of type 'NoneType' has no len()" in str(typ_err.value)


def test_linear_regression_missing_multiplicity_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
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
    instance = LinearRegressionOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(),
                     {'df': df})
        assert "multiplicity" in str(key_err.value)
