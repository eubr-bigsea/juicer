from tests.scikit_learn import util
from juicer.scikit_learn.clustering_operation import ClusteringModelOperation
from sklearn.cluster import *
from tests.scikit_learn.util import get_X_train_data, get_label_data
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# ClusteringModel
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_clustering_model_success():
    # Not working
    # AgglomerativeClustering
    # DBSCAN
    # FeatureAgglomeration
    # OPTICS()
    # SpectralClustering()
    # SpectralBiclustering()
    # SpectralCoclustering()
    models = [
        'AffinityPropagation()',

        'Birch()',

        'KMeans()',
        'MiniBatchKMeans()',
        'MeanShift()',

    ]

    for mod in models:
        df = util.iris(['sepallength', 'sepalwidth'], size=10)
        test_df = df.copy()

        arguments = {
            f'parameters': {'features': ['sepallength', 'sepalwidth'],
                            'multiplicity': {'train input data': 0}},
            'named_inputs': {
                'train input data': 'df',
                'algorithm': mod
            },
            'named_outputs': {
                'output data': 'out',
                'model': 'model'
            }
        }
        instance = ClusteringModelOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df': df})

        X = get_X_train_data(df, ['sepallength', 'sepalwidth'])
        var = eval(mod).fit(X)

        y = var.predict(X)
        test_out = test_df
        test_out['prediction'] = y

        if mod in [
            'AffinityPropagation()',
            'Birch()',
            'DBSCAN()',
            'MeanShift()'
        ]:
            assert result['out'].equals(test_out)
        else:
            assert not result['out'].equals(test_out)


def test_clustering_model_prediction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'prediction': 'success'},
        'named_inputs': {
            'train input data': 'df',
            'algorithm': 'KMeans()'
        },
        'named_outputs': {
            'output data': 'out',
            'model': 'model'
        }
    }
    instance = ClusteringModelOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    assert result['out'].columns[2] == 'success'


def test_clustering_model_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
            'algorithm': 'KMeans()'
        },
        'named_outputs': {
        }
    }
    instance = ClusteringModelOperation(**arguments)
    assert instance.generate_code() is None


def test_clustering_model_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out',
            'model': 'model'
        }
    }
    instance = ClusteringModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_clustering_model_missing_features_param_fail():
    arguments = {
        'parameters': {'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
            'algorithm': 'KMeans()'
        },
        'named_outputs': {
            'output data': 'out',
            'model': 'model'
        }
    }
    with pytest.raises(ValueError) as val_err:
        ClusteringModelOperation(**arguments)
    assert "Parameter 'features' must be informed for task" in \
           str(val_err)
