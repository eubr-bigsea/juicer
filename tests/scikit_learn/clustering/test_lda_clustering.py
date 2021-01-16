from tests.scikit_learn import util
from juicer.scikit_learn.clustering_operation import LdaClusteringOperation
from juicer.scikit_learn.util import get_X_train_data, get_label_data
from sklearn.decomposition import LatentDirichletAllocation
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# LdaClustering
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_lda_clustering_success():
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
    instance = LdaClusteringOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    model_1 = LatentDirichletAllocation(n_components=10,
                                        doc_topic_prior=None,
                                        topic_word_prior=None,
                                        learning_method='online', max_iter=10,
                                        random_state=None)

    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    model_1.fit(X_train)

    test_out = test_df
    test_out['prediction'] = model_1.transform(X_train).tolist()
    for idx in result['out'].index:
        for col in result['out'].columns:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 1.0e-03, 1.0e+03)


def test_lda_clustering_number_of_topics_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'number_of_topics': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LdaClusteringOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    model_1 = LatentDirichletAllocation(n_components=2,
                                        doc_topic_prior=None,
                                        topic_word_prior=None,
                                        learning_method='online', max_iter=10,
                                        random_state=None)

    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    model_1.fit(X_train)

    test_out = test_df
    test_out['prediction'] = model_1.transform(X_train).tolist()
    for idx in result['out'].index:
        for col in result['out'].columns:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 1.0e-03, 1.0e+03)


def test_lda_clustering_doc_topic_pior_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'doc_topic_pior': 0.2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LdaClusteringOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    model_1 = LatentDirichletAllocation(n_components=10,
                                        doc_topic_prior=0.2,
                                        topic_word_prior=None,
                                        learning_method='online', max_iter=10,
                                        random_state=None)

    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    model_1.fit(X_train)

    test_out = test_df
    test_out['prediction'] = model_1.transform(X_train).tolist()
    for idx in result['out'].index:
        for col in result['out'].columns:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 1.0e-03, 1.0e+03)


def test_lda_clustering_topic_word_prior_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'topic_word_prior': 0.5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LdaClusteringOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    model_1 = LatentDirichletAllocation(n_components=10,
                                        doc_topic_prior=None,
                                        topic_word_prior=0.5,
                                        learning_method='online', max_iter=10,
                                        random_state=None)

    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    model_1.fit(X_train)

    test_out = test_df
    test_out['prediction'] = model_1.transform(X_train).tolist()
    for idx in result['out'].index:
        for col in result['out'].columns:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 1.0e-03, 1.0e+03)


def test_lda_clustering_learning_method_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'learning_method': 'batch'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LdaClusteringOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    model_1 = LatentDirichletAllocation(n_components=10,
                                        doc_topic_prior=None,
                                        topic_word_prior=None,
                                        learning_method='batch', max_iter=10,
                                        random_state=None)

    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    model_1.fit(X_train)

    test_out = test_df
    test_out['prediction'] = model_1.transform(X_train).tolist()
    for idx in result['out'].index:
        for col in result['out'].columns:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 1.0e-03, 1.0e+03)


def test_lda_clustering_max_iter_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'max_iter': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LdaClusteringOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    model_1 = LatentDirichletAllocation(n_components=10,
                                        doc_topic_prior=None,
                                        topic_word_prior=None,
                                        learning_method='online', max_iter=2,
                                        random_state=None)

    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    model_1.fit(X_train)

    test_out = test_df
    test_out['prediction'] = model_1.transform(X_train).tolist()
    for idx in result['out'].index:
        for col in result['out'].columns:
            assert result['out'].loc[idx, col] == pytest.approx(
                test_out.loc[idx, col], 1.0e-03, 1.0e+03)


def test_lda_clustering_seed_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'seed': 2002},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LdaClusteringOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    model_1 = LatentDirichletAllocation(n_components=10,
                                        doc_topic_prior=None,
                                        topic_word_prior=None,
                                        learning_method='online', max_iter=10,
                                        random_state=2002)

    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    model_1.fit(X_train)

    test_out = test_df
    test_out['prediction'] = model_1.transform(X_train).tolist()
    assert result['out'].equals(test_out)


def test_lda_clustering_prediction_param_success():
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
    instance = LdaClusteringOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[2] == 'success'


def test_lda_clustering_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = LdaClusteringOperation(**arguments)
    assert instance.generate_code() is None


def test_lda_clustering_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = LdaClusteringOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_lda_clustering_invalid_learning_method_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'learning_method': 'invalid'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        LdaClusteringOperation(**arguments)
    assert "Invalid optimizer value 'invalid' for class" in str(val_err.value)


def test_lda_clustering_invalid_n_clusters_max_iter_params_fail():
    pars = ['number_of_topics',
            'max_iter']
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
            LdaClusteringOperation(**arguments)
        assert f"Parameter '{val}' must be x>0 for task" in str(val_err.value)


def test_lda_clustering_missing_features_param_fail():
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
        LdaClusteringOperation(**arguments)
    assert "Parameters 'features' must be informed for task" in str(
        val_err.value)
