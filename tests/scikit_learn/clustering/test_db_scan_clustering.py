from tests.scikit_learn import util
from juicer.scikit_learn.clustering_operation import DBSCANClusteringOperation
from tests.scikit_learn.util import get_X_train_data
from sklearn.cluster import DBSCAN
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# DBSCANClustering
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_db_scan_clustering_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

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
    instance = DBSCANClusteringOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_out, ['sepallength', 'sepalwidth'])
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
    test_out['cluster'] = dbscan.fit_predict(X_train)

    assert result['out'].equals(test_out)


def test_db_scan_clustering_eps_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'eps': 0.2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DBSCANClusteringOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_out, ['sepallength', 'sepalwidth'])
    dbscan = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
    test_out['cluster'] = dbscan.fit_predict(X_train)

    assert result['out'].equals(test_out)


def test_db_scan_clustering_min_samples_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'min_samples': 10},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DBSCANClusteringOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_out, ['sepallength', 'sepalwidth'])
    dbscan = DBSCAN(eps=0.5, min_samples=10, metric='euclidean')
    test_out['cluster'] = dbscan.fit_predict(X_train)

    assert result['out'].equals(test_out)


def test_db_scan_clustering_prediction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'prediction': 'success'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DBSCANClusteringOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})
    assert result['out'].columns[2] == 'success'


def test_db_scan_clustering_metric_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'metric': 'manhattan'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DBSCANClusteringOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    X_train = get_X_train_data(test_out, ['sepallength', 'sepalwidth'])
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='manhattan')
    test_out['cluster'] = dbscan.fit_predict(X_train)

    assert result['out'].equals(test_out)


def test_db_scan_clustering_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = DBSCANClusteringOperation(**arguments)
    assert instance.generate_code() is None


def test_db_scan_clustering_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = DBSCANClusteringOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_db_scan_clustering_missing_features_param_fail():
    arguments = {
        'parameters': {'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        DBSCANClusteringOperation(**arguments)
    assert "Parameter 'features' must be informed for task" in str(val_err.value)


def test_db_scan_clustering_invalid_eps_and_min_samples_params_fail():
    pars = [
        'min_samples',
        'eps'
    ]
    for val in pars:
        arguments = {
            'parameters': {f'features': ['sepallength', 'sepalwidth'],
                           'multiplicity': {'train input data': 0},
                           val: -1},
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        with pytest.raises(ValueError) as val_err:
            DBSCANClusteringOperation(**arguments)
        assert f"Parameter '{val}' must be x>0 for task" in str(val_err.value)
