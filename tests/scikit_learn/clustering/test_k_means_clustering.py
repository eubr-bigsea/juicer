from tests.scikit_learn import util
from juicer.scikit_learn.clustering_operation import KMeansClusteringOperation
from tests.scikit_learn.util import get_X_train_data, get_label_data
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# KMeansClustering
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_k_means_clustering_success():
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
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = KMeans(n_clusters=8, init='k-means++', max_iter=100,
                     tol=0.0001, random_state=None, n_init=10,
                     n_jobs=None, algorithm='auto')
    test_out['prediction'] = model_1.fit_predict(X_train)

    assert not result['out'].equals(test_out)


def test_k_means_clustering_n_clusters_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'n_clusters': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = KMeans(n_clusters=2, init='k-means++', max_iter=100,
                     tol=0.0001, random_state=None, n_init=10,
                     n_jobs=None, algorithm='auto')
    test_out['prediction'] = model_1.fit_predict(X_train)

    for idx in result['out'].index:
        for col in result['out'].columns:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 1.0e-12, 1.0e+12
            )


def test_k_means_clustering_init_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'init': 'random'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = KMeans(n_clusters=8, init='random', max_iter=100,
                     tol=0.0001, random_state=None, n_init=10,
                     n_jobs=None, algorithm='auto')
    test_out['prediction'] = model_1.fit_predict(X_train)

    assert not result['out'].equals(test_out)


def test_k_means_clustering_max_iter_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'max_iter': 50},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = KMeans(n_clusters=8, init='k-means++', max_iter=50,
                     tol=0.0001, random_state=None, n_init=10,
                     n_jobs=None, algorithm='auto')
    test_out['prediction'] = model_1.fit_predict(X_train)

    assert not result['out'].equals(test_out)


def test_k_means_clustering_tolerance_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'tolerance': 0.1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = KMeans(n_clusters=8, init='k-means++', max_iter=100,
                     tol=0.1, random_state=None, n_init=10,
                     n_jobs=None, algorithm='auto')
    test_out['prediction'] = model_1.fit_predict(X_train)

    assert not result['out'].equals(test_out)


def test_k_means_clustering_mini_batch_k_mean_type_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'type': 'Mini-Batch K-Means'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = test_df[['sepallength', 'sepalwidth']].to_numpy().tolist()
    model_1 = MiniBatchKMeans(n_clusters=8, init='k-means++',
                              max_iter=100, tol=0.0,
                              random_state=None, n_init=3,
                              max_no_improvement=10,
                              batch_size=100)
    test_out['prediction'] = model_1.fit_predict(X_train)

    assert not result['out'].equals(test_out)


def test_k_means_clustering_random_state_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'random_state': 2002},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = KMeans(n_clusters=8, init='k-means++', max_iter=100,
                     tol=0.0001, random_state=2002, n_init=10,
                     n_jobs=None, algorithm='auto')
    test_out['prediction'] = model_1.fit_predict(X_train)

    assert result['out'].equals(test_out)


def test_k_means_clustering_n_init_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'n_init': 5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = KMeans(n_clusters=8, init='k-means++', max_iter=100,
                     tol=0.0001, random_state=None, n_init=5,
                     n_jobs=None, algorithm='auto')
    test_out['prediction'] = model_1.fit_predict(X_train)

    assert not result['out'].equals(test_out)


def test_k_means_clustering_n_init_mb_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'type': 'Mini-Batch K-Means',
                       'n_init_mb': 6},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = test_df[['sepallength', 'sepalwidth']].to_numpy().tolist()
    model_1 = MiniBatchKMeans(n_clusters=8, init='k-means++',
                              max_iter=100, tol=0.0,
                              random_state=None, n_init=6,
                              max_no_improvement=10,
                              batch_size=100)
    test_out['prediction'] = model_1.fit_predict(X_train)

    assert not result['out'].equals(test_out)


def test_k_means_clustering_n_jobs_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'n_jobs': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = KMeans(n_clusters=8, init='k-means++', max_iter=100,
                     tol=0.0001, random_state=None, n_init=10,
                     n_jobs=2, algorithm='auto')
    test_out['prediction'] = model_1.fit_predict(X_train)

    assert not result['out'].equals(test_out)


def test_k_means_clustering_algorithm_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'algorithm': 'full'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = KMeans(n_clusters=8, init='k-means++', max_iter=100,
                     tol=0.0001, random_state=None, n_init=10,
                     n_jobs=None, algorithm='full')
    test_out['prediction'] = model_1.fit_predict(X_train)

    assert not result['out'].equals(test_out)


def test_k_means_clustering_batch_size_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'type': 'Mini-Batch K-Means',
                       'batch_size': 25},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = test_df[['sepallength', 'sepalwidth']].to_numpy().tolist()
    model_1 = MiniBatchKMeans(n_clusters=8, init='k-means++',
                              max_iter=100, tol=0.0,
                              random_state=None, n_init=3,
                              max_no_improvement=10,
                              batch_size=25)
    test_out['prediction'] = model_1.fit_predict(X_train)

    assert not result['out'].equals(test_out)


def test_k_means_clustering_tol_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'type': 'Mini-Batch K-Means',
                       'tol': 0.2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = test_df[['sepallength', 'sepalwidth']].to_numpy().tolist()
    model_1 = MiniBatchKMeans(n_clusters=8, init='k-means++',
                              max_iter=100, tol=0.2,
                              random_state=None, n_init=3,
                              max_no_improvement=10,
                              batch_size=100)
    test_out['prediction'] = model_1.fit_predict(X_train)

    assert not result['out'].equals(test_out)


def test_k_means_clustering_max_no_improvement_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'type': 'Mini-Batch K-Means',
                       'max_no_improvement': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    test_out = test_df
    X_train = test_df[['sepallength', 'sepalwidth']].to_numpy().tolist()
    model_1 = MiniBatchKMeans(n_clusters=8, init='k-means++',
                              max_iter=100, tol=0.0,
                              random_state=None, n_init=3,
                              max_no_improvement=2,
                              batch_size=100)
    test_out['prediction'] = model_1.fit_predict(X_train)

    assert not result['out'].equals(test_out)


def test_k_means_clustering_prediction_param_success():
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
    instance = KMeansClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].columns[2] == 'success'


def test_k_means_clustering_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    assert instance.generate_code() is None


def test_k_means_clustering_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KMeansClusteringOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_k_means_clustering_invalid_n_clusters_max_iter_params_fail():
    pars = [
        'n_clusters',
        'max_iter'
    ]
    for val in pars:
        arguments = {
            'parameters': {'features': ['sepallength', 'sepalwidth'],
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
            KMeansClusteringOperation(**arguments)
        assert f"Parameter '{val}' must be x>0 for task" in \
               str(val_err.value)
