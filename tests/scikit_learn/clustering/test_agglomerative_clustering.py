from tests.scikit_learn import util
from juicer.scikit_learn.clustering_operation import \
    AgglomerativeClusteringOperation
from sklearn.cluster import AgglomerativeClustering
from tests.scikit_learn.util import get_X_train_data
import pytest
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# AgglomerativeClustering
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_agglomerative_clustering_success():
    df = util.iris(['petalwidth', 'petallength'], size=30)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['petalwidth', 'petallength'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AgglomerativeClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    X = get_X_train_data(test_out, ['petalwidth', 'petallength'])
    agg = AgglomerativeClustering(n_clusters=2,
                                  linkage='ward', affinity='euclidean')
    test_out['cluster'] = agg.fit_predict(X)

    assert result['out'].equals(test_out)


def test_agglomerative_clustering_alias_param_success():
    df = util.iris(['petalwidth', 'petallength'], size=30)
    arguments = {
        'parameters': {'attributes': ['petalwidth', 'petallength'],
                       'multiplicity': {'input data': 0},
                       'alias': 'success'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AgglomerativeClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})
    assert result['out'].columns[2] == 'success'


def test_agglomerative_clustering_number_of_clusters_param_success():
    df = util.iris(['petalwidth', 'petallength'], size=30)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['petalwidth', 'petallength'],
                       'multiplicity': {'input data': 0},
                       'number_of_clusters': 5},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AgglomerativeClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    X = get_X_train_data(test_out, ['petalwidth', 'petallength'])
    agg = AgglomerativeClustering(n_clusters=5,
                                  linkage='ward', affinity='euclidean')
    test_out['cluster'] = agg.fit_predict(X)

    assert result['out'].equals(test_out)


def test_agglomerative_clustering_affinity_and_linkage_params_success():
    df = util.iris(['petalwidth', 'petallength'], size=30)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['petalwidth', 'petallength'],
                       'multiplicity': {'input data': 0},
                       'linkage': 'single',
                       'affinity': 'l1'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AgglomerativeClusteringOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    X = get_X_train_data(test_out, ['petalwidth', 'petallength'])
    agg = AgglomerativeClustering(n_clusters=2,
                                  linkage='single', affinity='l1')
    test_out['cluster'] = agg.fit_predict(X)

    assert result['out'].equals(test_out)


def test_agglomerative_clustering_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['petalwidth', 'petallength'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = AgglomerativeClusteringOperation(**arguments)
    assert instance.generate_code() is None


def test_agglomerative_clustering_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['petalwidth', 'petallength'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = AgglomerativeClusteringOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_agglomerative_clustering_missing_attributes_param_fail():
    arguments = {
        'parameters': {'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        AgglomerativeClusteringOperation(**arguments)
    assert "Parameter 'attributes' must be informed for task" in str(val_err)


def test_agglomerative_clustering_invalid_number_of_clusters_param_fail():
    arguments = {
        'parameters': {'attributes': ['petalwidth', 'petallength'],
                       'multiplicity': {'input data': 0},
                       'number_of_clusters': -1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        AgglomerativeClusteringOperation(**arguments)
    assert "Parameter 'number_of_clusters' must be x>0 for task" in str(val_err)
