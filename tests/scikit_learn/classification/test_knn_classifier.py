from tests.scikit_learn import util
from juicer.scikit_learn.classification_operation import KNNClassifierOperation
from sklearn.neighbors import KNeighborsClassifier
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# KNNClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_knn_classifier_success():
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
    instance = KNNClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params=None, n_jobs=None)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_knn_classifier_n_neighbors_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'n_neighbors': 10},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KNNClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = KNeighborsClassifier(n_neighbors=10,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params=None, n_jobs=None)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_knn_classifier_weights_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'weights': 'distance'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KNNClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='distance', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params=None, n_jobs=None)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_knn_classifier_algorithm_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'algorithm': 'brute'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KNNClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='brute',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params=None, n_jobs=None)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_knn_classifier_leaf_size_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'leaf_size': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KNNClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=2, p=2, metric='minkowski',
                                   metric_params=None, n_jobs=None)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_knn_classifier_p_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'p': 4},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KNNClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=4, metric='minkowski',
                                   metric_params=None, n_jobs=None)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_knn_classifier_metric_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'metric': 'braycurtis'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KNNClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='braycurtis',
                                   metric_params=None, n_jobs=None)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_knn_classifier_metric_params_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'metric_params': {'1': '1'}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KNNClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params={'1': '1'}, n_jobs=None)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_knn_classifier_n_jobs_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'n_jobs': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KNNClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    model_1 = KNeighborsClassifier(n_neighbors=5,
                                   weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski',
                                   metric_params=None, n_jobs=2)
    assert str(result['model_1']) == str(model_1)
    assert not result['out'].equals(test_df)


def test_knn_classifier_prediction_param_success():
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
    instance = KNNClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[2] == 'success'


def test_knn_classifier_no_output_implies_no_code_success():
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
    instance = KNNClassifierOperation(**arguments)
    assert instance.generate_code() is None


def test_knn_classifier_missing_input_implies_no_code_success():
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
    instance = KNNClassifierOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_knn_classifier_invalid_invalid_n_neighbors_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'n_neighbors': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        KNNClassifierOperation(**arguments)
    assert "Parameter 'n_neighbors' must be x>0 for task" in str(val_err.value)


def test_knn_classifier_invalid_p_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepallength'],
                       'p': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        KNNClassifierOperation(**arguments)
    assert "Parameter 'p' must be x>1 when parameter 'metric_params'" \
           " is 'minkowski' for task" in str(val_err.value)
