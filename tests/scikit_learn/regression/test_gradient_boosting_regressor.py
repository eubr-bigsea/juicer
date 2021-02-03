from tests.scikit_learn import util
from juicer.scikit_learn.regression_operation import \
    GradientBoostingRegressorOperation
from sklearn.ensemble import GradientBoostingRegressor
from tests.scikit_learn.util import get_X_train_data, get_label_data
import pandas as pd
import pytest

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# GradientBoostingRegressor:
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_gradient_boosting_regressor_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
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
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_learning_rate_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'learning_rate': 2.0},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=2.0,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_n_estimators_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'n_estimators': 200},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=200, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_max_depth_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_depth': 6},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=6,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_min_split_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_samples_split': 6},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=6,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_min_samples_leaf_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_samples_leaf': 3},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=3,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_max_features_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_features': 'auto'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features='auto',
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_criterion_parm_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'criterion': 'mse'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_min_wieght_fraction_leaf_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_weight_fraction_leaf': 0.5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.5,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_max_leaf_nodes_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_leaf_nodes': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=2,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_min_impurity_decrease_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_impurity_decrease': 0.5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.5,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_random_state_param_success():
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
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=2002, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_verbose_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'verbose': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=1,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_prediction_param_success():
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
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[2] == 'success'


def test_gradient_boosting_regressor_loss_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'loss': 'huber'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='huber',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_subsample_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'subsample': 0.5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=0.5,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in test_out.columns:
        for idx in test_out.index:
            assert test_out.loc[idx, col] == pytest.approx(
                result['out'].loc[idx, col], 0.01)


def test_gradient_boosting_regressor_alpha_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'alpha': 0.6},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.6, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_cc_alpha_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'cc_alpha': 0.5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.5,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_validation_fraction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'validation_fraction': 0.3},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.3,
        n_iter_no_change=None, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_n_iter_no_change_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'n_iter_no_change': 5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=5, tol=0.0001
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    for col in test_out.columns:
        for idx in test_out.index:
            assert test_out.loc[idx, col] == pytest.approx(
                result['out'].loc[idx, col], 0.1)


def test_gradient_boosting_regressor_tol_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
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
    instance = GradientBoostingRegressorOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])

    model_1 = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100, subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        random_state=None, max_features=None,
        alpha=0.9, verbose=0,
        max_leaf_nodes=None,
        warm_start=False, ccp_alpha=0.0,
        validation_fraction=0.1,
        n_iter_no_change=None, tol=0.01
    )

    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_gradient_boosting_regressor_no_output_implies_no_code_success():
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
    instance = GradientBoostingRegressorOperation(**arguments)
    assert instance.generate_code() is None


def test_gradient_boosting_regressor_missing_input_implies_no_code_success():
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
    instance = GradientBoostingRegressorOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_gradient_boosting_regressor_various_invalid_params_fail():
    params = {
        'learning_rate': -1,
        'n_estimators': -1,
        'min_samples_split': -1,
        'min_samples_leaf': -1,
        'max_depth': -1,
    }

    for val in params:
        arguments = {
            f'parameters': {'features': ['sepallength', 'sepalwidth'],
                            'multiplicity': {'train input data': 0},
                            'label': ['sepallength'],
                            val: params[val]
                            },
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        with pytest.raises(ValueError) as val_err:
            GradientBoostingRegressorOperation(**arguments)
        assert f"Parameter '{val}' must be x>0 for task" in str(val_err.value)


def test_gradient_boosting_regressor_invalid_random_state_n_iter_params_fail():
    params = {
        'random_state': -1,
        'n_iter_no_change': -1
    }

    for val in params:
        arguments = {
            f'parameters': {'features': ['sepallength', 'sepalwidth'],
                            'multiplicity': {'train input data': 0},
                            'label': ['sepallength'],
                            val: params[val]
                            },
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        with pytest.raises(ValueError) as val_err:
            GradientBoostingRegressorOperation(**arguments)
        assert f"Parameter '{val}' must be x >= 0 or None for task" in str(
            val_err.value)


def test_gradient_boosting_regressor_cc_alpha_min_impurity_params_fail():
    params = {
        'cc_alpha': -1,
        'min_impurity_decrease': -1
    }

    for val in params:
        arguments = {
            f'parameters': {'features': ['sepallength', 'sepalwidth'],
                            'multiplicity': {'train input data': 0},
                            'label': ['sepallength'],
                            val: params[val]
                            },
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        with pytest.raises(ValueError) as val_err:
            GradientBoostingRegressorOperation(**arguments)
        assert f"Parameter '{val}' must be x >= 0 for task" in str(val_err.value)


def test_gradient_boosting_regressor_invalid_validation_fraction_param_fail():
    arguments = {
        f'parameters': {'features': ['sepallength', 'sepalwidth'],
                        'multiplicity': {'train input data': 0},
                        'label': ['sepallength'],
                        'validation_fraction': -1
                        },
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        GradientBoostingRegressorOperation(**arguments)
    assert "Parameter 'validation_fraction' must be 0 <= x <= 1 for task" in str(
        val_err.value)


def test_gradient_boosting_regressor_invalid_subsample_param_fail():
    arguments = {
        f'parameters': {'features': ['sepallength', 'sepalwidth'],
                        'multiplicity': {'train input data': 0},
                        'label': ['sepallength'],
                        'subsample': -1
                        },
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        GradientBoostingRegressorOperation(**arguments)
    assert "Parameter 'subsample' must be 0 < x <= 1 for task" in str(
        val_err.value)


def test_gradient_boosting_regressor_invalid_min_wight_fraction_leaf_param_fail():
    arguments = {
        f'parameters': {'features': ['sepallength', 'sepalwidth'],
                        'multiplicity': {'train input data': 0},
                        'label': ['sepallength'],
                        'min_weight_fraction_leaf': -1
                        },
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        GradientBoostingRegressorOperation(**arguments)
    assert "Parameter 'min_weight_fraction_leaf' must be 0 <= x <= 0.5 for task" \
           in str(val_err.value)
