from tests.scikit_learn import util
from juicer.scikit_learn.classification_operation import GBTClassifierOperation
from sklearn.ensemble import GradientBoostingClassifier
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# GBTClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_gbt_classifier_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_learning_rate_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'learning_rate': 0.3},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.3,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_n_estimators_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'n_estimators': 50},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=50, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_max_depth_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_depth': 5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=5, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_min_samples_split_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_samples_split': 5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=5,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_min_samples_leaf_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=3,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_loss_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'loss': 'exponential'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='exponential',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_random_state_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=2002, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_subsample_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=0.5,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_criterion_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'criterion': 'mae'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='mae',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_min_weight_fraction_leaf_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.5,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_min_impurity_decrease_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_impurity_decrease': 0.2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.2, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_init_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'init': '"zero"'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init='zero',
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_max_features_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features='auto',
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_max_leaf_nodes_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=2, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_validation_fraction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'validation_fraction': 0.2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.2,
                                         n_iter_no_change=None, tol=0.0001)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_n_iter_no_change_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petallength', 'petalwidth'], size=150)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'n_iter_no_change': 4},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=4, tol=0.0001)

    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_tol_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth', ],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'tol': 0.1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = GradientBoostingClassifier(loss='deviance',
                                         learning_rate=0.1,
                                         n_estimators=100, min_samples_split=2,
                                         max_depth=3, min_samples_leaf=1,
                                         random_state=None, subsample=1.0,
                                         criterion='friedman_mse',
                                         min_weight_fraction_leaf=0.0,
                                         min_impurity_decrease=0.0, init=None,
                                         max_features=None,
                                         max_leaf_nodes=None, warm_start=False,
                                         validation_fraction=0.1,
                                         n_iter_no_change=None, tol=0.1)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_gbt_classifier_prediction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    instance = GBTClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[2] == 'success'


def test_gbt_classifier_no_output_implies_no_code_success():
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
    instance = GBTClassifierOperation(**arguments)
    assert instance.generate_code() is None


def test_gbt_classifier_missing_input_implies_no_code_success():
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
    instance = GBTClassifierOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_gbt_classifier_multiple_invalid_params_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    pars = [
        'min_samples_split',
        'min_samples_leaf',
        'learning_rate',
        'n_estimators',
        'max_depth'
    ]
    for arg in pars:
        arguments = {
            'parameters': {'features': ['sepallength', 'sepalwidth'],
                           'multiplicity': {'train input data': 0},
                           'label': ['sepallength'],
                           arg: -1},
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        with pytest.raises(ValueError) as val_err:
            GBTClassifierOperation(**arguments)
        assert f"Parameter '{arg}' must be x>0 for task" in str(val_err.value)


def test_gbt_classifier_invalid_max_leafs_nodes_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_leaf_nodes': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        GBTClassifierOperation(**arguments)
    assert "Parameter 'max_leaf_nodes' must be None or x > 1 for task" in str(
        val_err.value)


def test_gbt_classifier_invalid_n_iter_no_change_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'n_iter_no_change': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        GBTClassifierOperation(**arguments)
    assert "Parameter 'n_iter_no_change' must be None or x > 0 for task" in str(
        val_err.value)


def test_gbt_classifier_invalid_validation_fraction_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'validation_fraction': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        GBTClassifierOperation(**arguments)
    assert "Parameter 'validation_fraction' must be 0 <= x =< 1 for task" in \
           str(val_err.value)


def test_gbt_classifier_invalid_subsample_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'subsample': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        GBTClassifierOperation(**arguments)
    assert "Parameter 'subsample' must be 0 < x =< 1 for task" in \
           str(val_err.value)


def test_gbt_classifier_invalid_min_weight_fraction_leaf_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_weight_fraction_leaf': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        GBTClassifierOperation(**arguments)
    assert "Parameter 'min_weight_fraction_leaf' must be 0.0" \
           " <= x =< 0.5 for task" in str(val_err.value)
