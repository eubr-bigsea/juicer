from tests.scikit_learn import util
from juicer.scikit_learn.classification_operation import \
    DecisionTreeClassifierOperation, DecisionTreeClassifierModelOperation
from sklearn.tree import DecisionTreeClassifier
from tests.scikit_learn.util import get_label_data, get_X_train_data
import numpy as np
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# DecisionTreeClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_decision_tree_classifier_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
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
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
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
                                     class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_max_depth_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
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
                                     class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_min_samples_split_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
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
                                     class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_min_samples_leaf_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
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
                                     class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_min_weight_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
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
                                     class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_random_state_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
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
                                     class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_criterion_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
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
                                     class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_splitter_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
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
                                     class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_max_features_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
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
                                     class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_max_leafs_nodes_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     random_state=None, criterion='gini',
                                     splitter='best', max_features=None,
                                     max_leaf_nodes=2,
                                     min_impurity_decrease=0.0,
                                     class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_min_impurity_decrease_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_impurity_decrease': 2.0},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepallength'])
    y = np.reshape(y, len(y))
    model_1 = DecisionTreeClassifier(max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     random_state=None, criterion='gini',
                                     splitter='best', max_features=None,
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=2.0,
                                     class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_class_weight_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'class_weight': '"balanced"'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
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
                                     class_weight="balanced")
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_presort_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'presort': True},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
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
                                     class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_decision_tree_classifier_prediction_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[2] == 'success'


def test_decision_tree_classifier_no_output_implies_no_code_success():
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
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


def test_decision_tree_classifier_missing_input_implies_no_code_success():
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
    util.add_minimum_ml_args(arguments)
    instance = DecisionTreeClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_decision_tree_classifier_invalid_min_samples_split_leaf_params_fail():
    pars = ['min_samples_split',
            'min_samples_leaf']
    for val in pars:
        arguments = {
            'parameters': {'features': ['sepallength', 'sepalwidth'],
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
        util.add_minimum_ml_args(arguments)
        with pytest.raises(ValueError) as val_err:
            DecisionTreeClassifierModelOperation(**arguments)
        assert f"Parameter '{val}' must be x>0 for task" \
               f" {DecisionTreeClassifierOperation}" in str(val_err)


def test_decision_tree_classifier_invalid_min_weight_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_weight': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        DecisionTreeClassifierModelOperation(**arguments)
    assert f"Parameter 'min_weight' must be x>=0 or x<=0.5 for task" \
           f" {DecisionTreeClassifierOperation}" in str(val_err)


def test_decision_tree_classifier_invalid_max_depth_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'max_depth': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        DecisionTreeClassifierModelOperation(**arguments)
    assert f"Parameter 'max_depth' must be x>0 for task " \
           f"{DecisionTreeClassifierOperation}" in str(val_err)


def test_decision_tree_classifier_min_samples_split_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_samples_split': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        DecisionTreeClassifierModelOperation(**arguments)
    assert f"Parameter 'min_samples_split' must be x>0 for task " \
           f"{DecisionTreeClassifierOperation}" in str(val_err)


def test_decision_tree_classifier_min_impurity_decrease_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'min_impurity_decrease': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        DecisionTreeClassifierModelOperation(**arguments)
    assert f"Parameter 'min_impurity_decrease' must be x>=0 for task " \
           f"{DecisionTreeClassifierOperation}" in str(val_err)
