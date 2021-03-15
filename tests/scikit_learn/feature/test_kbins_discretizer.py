from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation \
    import KBinsDiscretizerOperation
from sklearn.preprocessing import KBinsDiscretizer
from textwrap import dedent
from tests.scikit_learn.util import get_X_train_data
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# KBinsDiscretizer
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_kbins_discretizer_alias_n_quantiles_params_success():
    df = util.iris(['sepallength'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attribute': ['sepallength'],
                       'n_quantiles': 2,
                       'multiplicity': {'input data': 0},
                       'alias': "success"
                       },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KBinsDiscretizerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    X_train = util.get_X_train_data(test_df, ['sepallength'])
    model_1 = KBinsDiscretizer(n_bins=2,
                               encode='ordinal',
                               strategy='quantile')

    test_df["success"] = model_1.fit_transform(X_train).flatten().tolist()
    assert result['out'].columns[1] == 'success'
    assert result['out'].equals(test_df)
    assert dedent("""
    out = df
    model_1 = KBinsDiscretizer(n_bins=2, 
        encode='ordinal', strategy='quantile')
    X_train = get_X_train_data(df, ['sepallength'])

    values = model_1.fit_transform(X_train)

    out = pd.concat([df, 
        pd.DataFrame(values, columns=['success'])],
        ignore_index=False, axis=1)
    """)


def test_kbins_discretizer_uniform_output_distribution_param_success():
    df = util.iris(['sepallength'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {
            'attribute': ['sepallength'],
            'multiplicity': {'input data': 0},
            'output_distribution': 'uniform'

        },
        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KBinsDiscretizerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    model_1 = KBinsDiscretizer(n_bins=5,
                               encode='ordinal', strategy='uniform')
    X_train = get_X_train_data(test_df, ['sepallength'])
    test_df['sepallength_disc'] = model_1.fit_transform(
        X_train).flatten().tolist()

    assert result['out'].equals(test_df)
    assert dedent("""
    out = df
    model_1 = KBinsDiscretizer(n_bins=5, 
        encode='ordinal', strategy='uniform')
    X_train = get_X_train_data(df, ['sepallength'])

    values = model_1.fit_transform(X_train)

    out = pd.concat([df, 
        pd.DataFrame(values, columns=['sepallength_disc'])],
        ignore_index=False, axis=1)
    """)


def test_kbins_discretizer_kmeans_output_distribution_param_success():
    df = util.iris(['sepallength'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {
            'attribute': ['sepallength'],
            'multiplicity': {'input data': 0},
            'output_distribution': 'kmeans',
            'n_quantiles': 2

        },
        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KBinsDiscretizerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    model_1 = KBinsDiscretizer(n_bins=2,
                               encode='ordinal', strategy='kmeans')
    X_train = get_X_train_data(test_df, ['sepallength'])
    test_df['sepallength_disc'] = model_1.fit_transform(
        X_train).flatten().tolist()

    assert result['out'].equals(test_df)
    assert dedent("""
    out = df
    model_1 = KBinsDiscretizer(n_bins=2, 
        encode='ordinal', strategy='kmeans')
    X_train = get_X_train_data(df, ['sepallength'])

    values = model_1.fit_transform(X_train)

    out = pd.concat([df, 
        pd.DataFrame(values, columns=['sepallength_disc'])],
        ignore_index=False, axis=1)
    """)


def test_kbins_discretizer_quantile_output_distribution_param_success():
    df = util.iris(['sepallength'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {
            'attribute': ['sepallength'],
            'multiplicity': {'input data': 0},
            'output_distribution': 'quantile'

        },
        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KBinsDiscretizerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    model_1 = KBinsDiscretizer(n_bins=5,
                               encode='ordinal', strategy='quantile')
    X_train = get_X_train_data(test_df, ['sepallength'])
    test_df['sepallength_disc'] = model_1.fit_transform(
        X_train).flatten().tolist()

    assert result['out'].equals(test_df)
    assert dedent("""
    out = df
    model_1 = KBinsDiscretizer(n_bins=5, 
        encode='ordinal', strategy='quantile')
    X_train = get_X_train_data(df, ['sepallength'])

    values = model_1.fit_transform(X_train)

    out = pd.concat([df, 
        pd.DataFrame(values, columns=['sepallength_disc'])],
        ignore_index=False, axis=1)
    """)


def test_kbins_discretizer_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attribute': ['sepallength'],
                       'n_quantiles': 2,
                       'multiplicity': {'input data': 0},
                       'alias': "success"
                       },
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = KBinsDiscretizerOperation(**arguments)
    assert instance.generate_code() is None


def test_kbins_discretizer_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attribute': ['sepallength'],
                       'n_quantiles': 2,
                       'multiplicity': {'input data': 0},
                       'alias': "success"
                       },
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = KBinsDiscretizerOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_kbins_discretizer_invalid_n_quantiles_param_fail():
    arguments = {
        'parameters': {
            'attribute': ['sepallength'],
            'multiplicity': {'input data': 0},
            'n_quantiles': -1
        },
        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        KBinsDiscretizerOperation(**arguments)
    assert f"Parameter 'n_quantiles' must be x>0 for task" \
           f" {KBinsDiscretizerOperation}" in str(val_err.value)


def test_kbins_discretizer_missing_attribute_param_fail():
    arguments = {
        'parameters': {
            'multiplicity': {'input data': 0},
        },
        'named_inputs': {
            'input data': 'df'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        KBinsDiscretizerOperation(**arguments)
    assert f"Parameters 'attributes' must be informed for task" \
           f" {KBinsDiscretizerOperation}" in str(val_err.value)
