from tests.scikit_learn import util
from juicer.scikit_learn.regression_operation import \
    RandomForestRegressorOperation
from sklearn.ensemble import RandomForestRegressor
from tests.scikit_learn.util import get_X_train_data, get_label_data
import pytest
import pandas as pd

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# RandomForestRegressor:
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_random_forest_regressor_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
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
    instance = RandomForestRegressorOperation(**arguments)

    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='auto',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=1, criterion='mse',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True, verbose=0, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_random_forest_regressor_n_estimators_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'n_estimators': 200},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=200,
        max_features='auto',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=1, criterion='mse',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True, verbose=0, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_random_forest_regressor_max_features_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'max_features': 'sqrt'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='sqrt',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=1, criterion='mse',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True, verbose=0, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_random_forest_regressor_max_depth_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'max_depth': 5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='auto',
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=1, criterion='mse',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True, verbose=0, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_random_forest_regressor_min_samples_split_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'min_samples_split': 6},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='auto',
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=1, criterion='mse',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True, verbose=0, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_random_forest_regressor_min_samples_leaf_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'min_samples_leaf': 8},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='auto',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=8,
        random_state=None,
        n_jobs=1, criterion='mse',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True, verbose=0, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_random_forest_regressor_criterion_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'criterion': 'mae'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='auto',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=1, criterion='mae',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True, verbose=0, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_random_forest_regressor_min_weight_fraction_leaf_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'min_weight_fraction_leaf': 0.5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='auto',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=1, criterion='mse',
        min_weight_fraction_leaf=0.5,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True, verbose=0, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_random_forest_regressor_max_leaf_nodes_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'max_leaf_nodes': 5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='auto',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=1, criterion='mse',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=5,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True, verbose=0, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_random_forest_regressor_min_impurity_decrease_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'min_impurity_decrease': 0.5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='auto',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=1, criterion='mse',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.5,
        bootstrap=True,
        oob_score=True, verbose=0, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_random_forest_regressor_bootstrap_and_oob_score_params_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'bootstrap': False,
                       'oob_score': False},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='auto',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=1, criterion='mse',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False, verbose=0, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_random_forest_regressor_n_jobs_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'n_jobs': 3},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='auto',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=3, criterion='mse',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True, verbose=0, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_random_forest_regressor_random_state_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'random_state': 2002},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='auto',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=2002,
        n_jobs=1, criterion='mse',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True, verbose=0, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_random_forest_regressor_verbose_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'verbose': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])

    model_1 = RandomForestRegressor(
        n_estimators=100,
        max_features='auto',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=1, criterion='mse',
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=True, verbose=2, warm_start=False
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in result['out'].columns:
        for idx in result['out'].index:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 0.1)


def test_random_forest_regressor_prediction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
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
    instance = RandomForestRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[2] == 'success'


def test_random_forest_regressor_no_output_implies_no_code_success():
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
    instance = RandomForestRegressorOperation(**arguments)
    assert instance.generate_code() is None


def test_random_forest_regressor_missing_input_implies_no_code_success():
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
    instance = RandomForestRegressorOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_random_forest_regressor_multiple_params_fail():
    params = ['n_estimators', 'min_samples_split',
              'min_samples_leaf']
    for val in params:
        arguments = {
            'parameters': {'features': ['sepallength', 'sepalwidth'],
                           'multiplicity': {'train input data': 0},
                           'label': ['sepalwidth'],
                           val: -1},
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        with pytest.raises(ValueError) as val_err:
            RandomForestRegressorOperation(**arguments)
        assert f"Parameter '{val}' must be x>0 for task" in str(
            val_err.value)


def test_random_forest_regressor_invalid_n_jobs_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'n_jobs': -3},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        RandomForestRegressorOperation(**arguments)
    assert "Parameter 'n_jobs' must be x >= -1 for task" in str(val_err.value)


def test_random_forest_regressor_invalid_max_depth_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'max_depth': -5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        RandomForestRegressorOperation(**arguments)
    assert "Parameter 'max_depth' must be x>0 or None for task" in str(
        val_err.value)


def test_random_forest_regressor_invalid_random_state_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'random_state': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        RandomForestRegressorOperation(**arguments)
    assert "Parameter 'random_state' must be x>=0 or None for task" in str(
        val_err.value)


def test_random_forest_regressor_invalid_min_weight_fraction_leaf_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'min_weight_fraction_leaf': 5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        RandomForestRegressorOperation(**arguments)
    assert "Parameter 'min_weight_fraction_leaf' must be x >= 0 and x" \
           " <= 0.5 for task" in str(val_err.value)


def test_random_forest_regressor_invalid_min_impurity_decrease_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'min_impurity_decrease': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        RandomForestRegressorOperation(**arguments)
    assert "Parameter 'min_impurity_decrease' must be x>=0 or None for task" \
           in str(val_err.value)

