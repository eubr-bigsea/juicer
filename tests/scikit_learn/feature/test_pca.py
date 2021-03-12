from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import PCAOperation
import pytest
import pandas as pd
from textwrap import dedent
from tests.scikit_learn.util import get_X_train_data
from sklearn.decomposition import PCA


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


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
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    pca = PCA(n_components=1)
    x_train = get_X_train_data(test_df, ['sepallength', 'petalwidth'])
    test_out['pca_feature'] = pca.fit_transform(x_train).tolist()
    assert result['out'].equals(test_out)
    assert dedent("""
    out = df
    pca = PCA(n_components=1)
    X_train = get_X_train_data(df, ['sepallength', 'petalwidth'])
    out['pca_feature'] = pca.fit_transform(X_train).tolist()
    """) == instance.generate_code()


def test_pca_two_n_components_success():
    df = util.iris(['sepallength',
                    'petalwidth'], size=2)
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
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    pca = PCA(n_components=2)
    x_train = get_X_train_data(test_df, ['sepallength', 'petalwidth'])
    test_out['pca_feature'] = pca.fit_transform(x_train).tolist()
    assert result['out'].equals(test_out)
    assert dedent("""
    out = df
    pca = PCA(n_components=2)
    X_train = get_X_train_data(df, ['sepallength', 'petalwidth'])
    out['pca_feature'] = pca.fit_transform(X_train).tolist()
    """) == instance.generate_code()


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
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    pca = PCA(n_components=1)
    x_train = get_X_train_data(test_df, ['sepallength', 'petalwidth'])
    test_out['success'] = pca.fit_transform(x_train).tolist()
    assert result['out'].equals(test_out)
    assert dedent("""
    out = df
    pca = PCA(n_components=1)
    X_train = get_X_train_data(df, ['sepallength', 'petalwidth'])
    out['success'] = pca.fit_transform(X_train).tolist()
    """) == instance.generate_code()


def test_pca_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attribute': ['sepallength', 'petalwidth'],
                       'k': 1,
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = PCAOperation(**arguments)
    assert instance.generate_code() is None


def test_pca_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attribute': ['sepallength', 'petalwidth'],
                       'k': 1,
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PCAOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_pca_missing_attributes_param_fail():
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
    with pytest.raises(ValueError) as val_err:
        PCAOperation(**arguments)
    assert f"Parameters 'attribute' must be informed for task" \
           f" {PCAOperation}" in str(val_err.value)


def test_pca_zero_n_parameters_fail():
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
    with pytest.raises(ValueError) as val_err:
        PCAOperation(**arguments)
    assert f"Parameter 'k' must be x>0 for task" \
           f" {PCAOperation}" in str(val_err.value)
