from tests.scikit_learn import util
from juicer.scikit_learn.clustering_operation import \
    GaussianMixtureClusteringOperation, GaussianMixtureClusteringModelOperation
from sklearn.mixture import GaussianMixture
from tests.scikit_learn.util import get_X_train_data
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# GaussianMixtureClustering
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_gaussian_mixture_clustering_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'random_state': 1,
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = GaussianMixture(n_components=3, max_iter=100, tol=0.001,
                              covariance_type='full', reg_covar=1e-06,
                              n_init=1, random_state=1)
    test_df['prediction'] = model_1.fit_predict(X_train)
    assert result['out'].equals(test_df)


def test_gaussian_mixture_clustering_max_iter_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'max_iter': 50, 'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = GaussianMixture(n_components=3, max_iter=50, tol=0.001,
                              covariance_type='full', reg_covar=1e-06,
                              n_init=1, random_state=1)
    test_df['prediction'] = model_1.fit_predict(X_train)
    assert result['out'].equals(test_df)


def test_gaussian_mixture_clustering_tol_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'tol': 0.1, 'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = GaussianMixture(n_components=3, max_iter=100, tol=0.1,
                              covariance_type='full', reg_covar=1e-06,
                              n_init=1, random_state=1)
    test_df['prediction'] = model_1.fit_predict(X_train)
    assert result['out'].equals(test_df)


def test_gaussian_mixture_clustering_prediction_param_success():
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[2] == 'success'


def test_gaussian_mixture_clustering_n_components_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'n_components': 2,
                       'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = GaussianMixture(n_components=2, max_iter=100, tol=0.001,
                              covariance_type='full', reg_covar=1e-06,
                              n_init=1, random_state=1)
    test_df['prediction'] = model_1.fit_predict(X_train)
    assert result['out'].equals(test_df)


def test_gaussian_mixture_clustering_covariance_type_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'covariance_type': 'tied', 'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = GaussianMixture(n_components=3, max_iter=100, tol=0.001,
                              covariance_type='tied', reg_covar=1e-06,
                              n_init=1, random_state=1)
    test_df['prediction'] = model_1.fit_predict(X_train)
    assert result['out'].equals(test_df)


def test_gaussian_mixture_clustering_reg_covar_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'reg_covar': 0.1,  'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = GaussianMixture(n_components=3, max_iter=100, tol=0.001,
                              covariance_type='full', reg_covar=0.1,
                              n_init=1, random_state=1)
    test_df['prediction'] = model_1.fit_predict(X_train)
    assert result['out'].equals(test_df)


def test_gaussian_mixture_clustering_n_init_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'n_init': 2, 'random_state': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = GaussianMixture(n_components=3, max_iter=100, tol=0.001,
                              covariance_type='full', reg_covar=1e-06,
                              n_init=2, random_state=1)
    test_df['prediction'] = model_1.fit_predict(X_train)

    assert result['out'].equals(test_df)


def test_gaussian_mixture_clustering_random_state_param_success():
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = GaussianMixture(n_components=3, max_iter=100, tol=0.001,
                              covariance_type='full', reg_covar=1e-06,
                              n_init=1, random_state=2002)
    test_df['prediction'] = model_1.fit_predict(X_train)
    assert result['out'].equals(test_df)


def test_gaussian_mixture_clustering_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    assert instance.generate_code() is None


def test_gaussian_mixture_clustering_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = GaussianMixtureClusteringModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_gaussian_mixture_clustering_invalid_n_comps_and_max_iter_params_fail():
    pars = [
        'max_iter',
        'n_components'
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
            GaussianMixtureClusteringModelOperation(**arguments)
        assert f"Parameter '{val}' must be x>0 for task" \
               f" {GaussianMixtureClusteringOperation}" in str(val_err.value)
