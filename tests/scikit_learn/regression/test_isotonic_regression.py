from tests.scikit_learn import util
from juicer.scikit_learn.regression_operation import IsotonicRegressionOperation
from sklearn.isotonic import IsotonicRegression
import pytest
import pandas as pd

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# IsotonicRegression:
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_isotonic_regression_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'isotonic': False},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = IsotonicRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    model_1 = IsotonicRegression(y_min=None, y_max=None, increasing=False,
                                 out_of_bounds='nan')

    test_out = test_df
    X_train = util.get_X_train_data(test_df, ['sepallength'])
    y = util.get_label_data(test_df, ['sepallength'])

    model_1.fit(X_train, y)

    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_isotonic_regression_isotonic_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'isotonic': True},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = IsotonicRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    model_1 = IsotonicRegression(y_min=None, y_max=None, increasing=True,
                                 out_of_bounds='nan')

    test_out = test_df
    X_train = util.get_X_train_data(test_df, ['sepallength'])
    y = util.get_label_data(test_df, ['sepallength'])

    model_1.fit(X_train, y)

    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_isotonic_regression_prediction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'features': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'isotonic': False,
                       'prediction': 'success', },
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = IsotonicRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[2] == 'success'


def test_isotonic_regression_y_min_y_max_params_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'isotonic': False,
                       'y_min': 2,
                       'y_max': 4},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = IsotonicRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    model_1 = IsotonicRegression(y_min=2.0, y_max=4.0, increasing=False,
                                 out_of_bounds='nan')

    test_out = test_df
    X_train = util.get_X_train_data(test_df, ['sepallength'])
    y = util.get_label_data(test_df, ['sepallength'])

    model_1.fit(X_train, y)

    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_isotonic_regression_out_of_bounds_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'isotonic': False,
                       'out_of_bounds': 'clip'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = IsotonicRegressionOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    model_1 = IsotonicRegression(y_min=None, y_max=None, increasing=False,
                                 out_of_bounds='clip')

    test_out = test_df
    X_train = util.get_X_train_data(test_df, ['sepallength'])
    y = util.get_label_data(test_df, ['sepallength'])

    model_1.fit(X_train, y)

    test_out['prediction'] = model_1.predict(X_train).tolist()

    assert result['out'].equals(test_out)


def test_isotonic_regression_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'isotonic': False},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = IsotonicRegressionOperation(**arguments)
    assert instance.generate_code() is None


def test_isotonic_regression_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'isotonic': False},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = IsotonicRegressionOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_isotonic_regression_invalid_size_features_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'isotonic': False},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        IsotonicRegressionOperation(**arguments)
    assert "Parameter 'features' must be x<2 for task" in \
           str(val_err.value)


def test_isotonic_regression_invalid_y_min_y_max_params_values_fail():
    arguments = {
        'parameters': {'features': ['sepallength'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'isotonic': False,
                       'y_min': 2,
                       'y_max': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        IsotonicRegressionOperation(**arguments)
    assert "Parameter 'y_min' must be less than or equal" \
           " to 'y_max' for task" in str(val_err.value)


def test_isotonic_regression_missing_multiplicity_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'features': ['sepallength'],
                       'label': ['sepallength'],
                       'isotonic': False},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    instance = IsotonicRegressionOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "multiplicity" in str(key_err.value)
