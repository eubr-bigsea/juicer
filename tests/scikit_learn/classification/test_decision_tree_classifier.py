from tests.scikit_learn import util
from juicer.scikit_learn.classification_operation import \
    DecisionTreeClassifierOperation
from sklearn.tree import DecisionTreeClassifier
from tests.scikit_learn.util import get_label_data, get_X_train_data
import numpy as np
import pandas as pd
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# DecisionTreeClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def xtest_decision_tree_classifier_success():
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
    instance = DecisionTreeClassifierOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     random_state=None, criterion='gini',
                                     splitter='best', max_features=None,
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     class_weight=None, presort=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_decision_tree_classifier_success():
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
    instance = DecisionTreeClassifierOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     random_state=None, criterion='gini',
                                     splitter='best', max_features=None,
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     class_weight=None, presort=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_decision_tree_classifier_max_depth_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_depth': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DecisionTreeClassifierOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=2,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     random_state=None, criterion='gini',
                                     splitter='best', max_features=None,
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     class_weight=None, presort=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_decision_tree_classifier_min_samples_split_param_success():
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
    instance = DecisionTreeClassifierOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=None,
                                     min_samples_split=5,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     random_state=None, criterion='gini',
                                     splitter='best', max_features=None,
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     class_weight=None, presort=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_decision_tree_classifier_min_samples_leaf_param_success():
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
    instance = DecisionTreeClassifierOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=3,
                                     min_weight_fraction_leaf=0.0,
                                     random_state=None, criterion='gini',
                                     splitter='best', max_features=None,
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     class_weight=None, presort=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_decision_tree_classifier_min_weight_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_weight': 0.5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DecisionTreeClassifierOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.5,
                                     random_state=None, criterion='gini',
                                     splitter='best', max_features=None,
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     class_weight=None, presort=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_decision_tree_classifier_random_state_param_success():
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
    instance = DecisionTreeClassifierOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     random_state=2002, criterion='gini',
                                     splitter='best', max_features=None,
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     class_weight=None, presort=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_decision_tree_classifier_criterion_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'criterion': 'entropy'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DecisionTreeClassifierOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     random_state=None, criterion='entropy',
                                     splitter='best', max_features=None,
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     class_weight=None, presort=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_decision_tree_classifier_splitter_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'splitter': 'random'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DecisionTreeClassifierOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     random_state=None, criterion='gini',
                                     splitter='random', max_features=None,
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     class_weight=None, presort=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_decision_tree_classifier_max_features_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_features': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DecisionTreeClassifierOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     random_state=None, criterion='gini',
                                     splitter='best', max_features=2,
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     class_weight=None, presort=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)


def test_decision_tree_classifier_max_leafs_nodes_param_success():
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
    instance = DecisionTreeClassifierOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     random_state=None, criterion='gini',
                                     splitter='best', max_features=None,
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     class_weight=None, presort=False)
    model_1.fit(X_train, y)
    test_out['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_out)
