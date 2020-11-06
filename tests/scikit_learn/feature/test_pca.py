from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import PCAOperation
import pytest
import pandas as pd
import numpy as np
from tests.scikit_learn.util import get_X_train_data
from sklearn.decomposition import PCA

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# PCA
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_pca_success():
    df = util.iris(['sepallength',
                    'petalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'attribute': ['sepallength', 'petalwidth'],
                       'k': 1,
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PCAOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    pca = PCA(n_components=1)
    x_train = get_X_train_data(test_df, ['sepallength', 'petalwidth'])
    test_out['pca_feature'] = pca.fit_transform(x_train).tolist()

    assert result['out'].equals(test_out)


def test_pca_two_n_components_success():
    df = util.iris(['sepallength',
                    'petalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'attribute': ['sepallength', 'petalwidth'],
                       'k': 2,
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PCAOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    pca = PCA(n_components=2)
    x_train = get_X_train_data(test_df, ['sepallength', 'petalwidth'])
    test_out['pca_feature'] = pca.fit_transform(x_train).tolist()

    assert result['out'].equals(test_out)


def test_pca_alias_param_success():
    df = util.iris(['sepallength',
                    'petalwidth'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'attribute': ['sepallength', 'petalwidth'],
                       'k': 1,
                       'multiplicity': {'input data': 0},
                       'alias': 'success'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PCAOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    pca = PCA(n_components=1)
    x_train = get_X_train_data(test_df, ['sepallength', 'petalwidth'])
    test_out['success'] = pca.fit_transform(x_train).tolist()

    assert result['out'].equals(test_out)


def test_pca_big_var_success():
    """
    Qual é o limite do n_components? Depende da quantidade de colunas/ linhas
    Montar uma função que testa todos os n_components de acordo com o critério acima?
    """
    df = util.iris(['sepallength',
                    'petalwidth',
                    'petallength',
                    'sepalwidth'], size=150)
    test_df = df.copy()

    arguments = {
        'parameters': {'attribute': ['sepallength', 'petalwidth', 'petallength',
                                     'sepalwidth'],
                       'k': 3,
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PCAOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    test_out = test_df
    pca = PCA(n_components=3)
    x_train = get_X_train_data(test_df,
                               ['sepallength', 'petalwidth', 'petallength',
                                'sepalwidth'])
    test_out['pca_feature'] = pca.fit_transform(x_train).tolist()

    assert result['out'].equals(test_out)


def test_pca_no_output_implies_no_code_success():
    df = util.iris(['sepallength',
                    'petalwidth'], size=10)
    arguments = {
        'parameters': {'attribute': ['sepallength', 'petalwidth'],
                       'k': 1,
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PCAOperation(**arguments)
    assert instance.generate_code() is None


def test_pca_missing_input_implies_no_code_success():
    df = util.iris(['sepallength',
                    'petalwidth'], size=10)
    arguments = {
        'parameters': {'attribute': ['sepallength', 'petalwidth'],
                       'k': 1,
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PCAOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_pca_missing_attributes_param_fail():
    df = util.iris(['sepallength',
                    'petalwidth'], size=10)

    arguments = {
        'parameters': {'k': 1,
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PCAOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})


def test_pca_missing_multiplicity_param_fail():
    df = util.iris(['sepallength',
                    'petalwidth'], size=10)

    arguments = {
        'parameters': {'attribute': ['sepallength', 'petalwidth'],
                       'k': 1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PCAOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})


def test_pca_zero_n_parameters_fail():
    df = util.iris(['sepallength',
                    'petalwidth'], size=10)

    arguments = {
        'parameters': {'attribute': ['sepallength', 'petalwidth'],
                       'k': 0,
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PCAOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

# def test_pca_string_fail():
#     df = util.iris(['sepallength',
#                     'petalwidth'], size=10)
#     df.iloc[0, 0] = 'string'
#     test_df = df.copy()
#
#     arguments = {
#         'parameters': {'attribute': ['sepallength', 'petalwidth'],
#                        'k': 1,
#                        'multiplicity': {'input data': 0}},
#         'named_inputs': {
#             'input data': 'df',
#         },
#         'named_outputs': {
#             'output data': 'out'
#         }
#     }
#     instance = PCAOperation(**arguments)
#     result = util.execute(instance.generate_code(),
#                           {'df': df})
#
#
# def test_pca_nan_fail():
#     df = util.iris(['sepallength',
#                     'petalwidth'], size=10)
#     df.iloc[0, 0] = np.NaN
#     test_df = df.copy()
#
#     arguments = {
#         'parameters': {'attribute': ['sepallength', 'petalwidth'],
#                        'k': 1,
#                        'multiplicity': {'input data': 0}},
#         'named_inputs': {
#             'input data': 'df',
#         },
#         'named_outputs': {
#             'output data': 'out'
#         }
#     }
#     instance = PCAOperation(**arguments)
#     result = util.execute(instance.generate_code(),
#                           {'df': df})
