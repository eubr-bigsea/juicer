from tests.scikit_learn import util
from juicer.scikit_learn.regression_operation import \
    GeneralizedLinearRegressionOperation
from sklearn.linear_model import LinearRegression
from tests.scikit_learn.util import get_X_train_data, get_label_data
import pandas as pd
import pytest

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# GeneralizedLinearRegression:
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_generalized_linear_regression_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features_atr': ['sepallength', 'sepalwidth'],
                       'labels': ['sepallength'],
                       'multiplicity': {'input data': 0},
                       'fit_intercept': 0,
                       'copy_X': 0},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GeneralizedLinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    model_1 = LinearRegression(
        fit_intercept=False, copy_X=False,
        n_jobs=1)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_generalized_linear_regression_fit_intercept_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features_atr': ['sepallength', 'sepalwidth'],
                       'labels': ['sepallength'],
                       'multiplicity': {'input data': 0},
                       'fit_intercept': 1,
                       'copy_X': 0},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GeneralizedLinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    model_1 = LinearRegression(
        fit_intercept=True, copy_X=False,
        n_jobs=1)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_generalized_linear_regression_normalize_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features_atr': ['sepallength', 'sepalwidth'],
                       'labels': ['sepallength'],
                       'multiplicity': {'input data': 0},
                       'fit_intercept': 1,
                       'normalize': 1,
                       'copy_X': 0},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GeneralizedLinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = LinearRegression(
        fit_intercept=True, normalize=True,
        copy_X=False, n_jobs=1)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(result['out'])


def test_generalized_linear_regression_copy_x_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features_atr': ['sepallength', 'sepalwidth'],
                       'labels': ['sepallength'],
                       'multiplicity': {'input data': 0},
                       'copy_X': 1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GeneralizedLinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = LinearRegression(
        fit_intercept=True, normalize=False,
        copy_X=True, n_jobs=1)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_generalized_linear_regression_n_jobs_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features_atr': ['sepallength', 'sepalwidth'],
                       'labels': ['sepallength'],
                       'multiplicity': {'input data': 0},
                       'n_jobs': 4},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GeneralizedLinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    model_1 = LinearRegression(
        fit_intercept=True, copy_X=False,
        n_jobs=4)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_generalized_linear_regression_alias_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'features_atr': ['sepallength', 'sepalwidth'],
                       'labels': ['sepallength'],
                       'multiplicity': {'input data': 0},
                       'alias': 'success'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GeneralizedLinearRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    assert result['out'].columns[2] == 'success'


def test_generalized_linear_regression_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features_atr': ['sepallength', 'sepalwidth'],
                       'labels': ['sepallength'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = GeneralizedLinearRegressionOperation(**arguments)
    assert instance.generate_code() is None


def test_generalized_linear_regression_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'features_atr': ['sepallength', 'sepalwidth'],
                       'labels': ['sepallength'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GeneralizedLinearRegressionOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_generalized_linear_regression_missing_labels_param_fail():
    arguments = {
        'parameters': {'features_atr': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        GeneralizedLinearRegressionOperation(**arguments)
    assert "Parameters 'labels' must be informed for task" in str(val_err.value)


def test_generalized_linear_regression_invalid_n_jobs_param_fail():
    arguments = {
        'parameters': {'features_atr': ['sepallength', 'sepalwidth'],
                       'labels': ['sepallength'],
                       'multiplicity': {'input data': 0},
                       'n_jobs': -2},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        GeneralizedLinearRegressionOperation(**arguments)
    assert "Parameter 'n_jobs' must be x>=-1 for task" in str(val_err.value)


def test_generalized_linear_regression_missing_multiplicity_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'features_atr': ['sepallength', 'sepalwidth'],
                       'labels': ['sepallength']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GeneralizedLinearRegressionOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(), {'df': df})
    assert 'multiplicity' in str(key_err.value)
