from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation \
    import QuantileDiscretizerOperation as QDO
from sklearn.preprocessing import KBinsDiscretizer
from tests.scikit_learn.util import get_X_train_data
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# QuantileDiscretizer
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_quantile_discretizer_alias_n_quantiles_params_success():
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
    instance = QDO(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    test_out = test_df
    model_1 = KBinsDiscretizer(n_bins=2,
                               encode='ordinal',
                               strategy='quantile')
    X_train = util.get_X_train_data(test_df, ['sepallength'])

    test_out["success"] = model_1.fit_transform(X_train).flatten().tolist()

    assert result['out'].columns[1] == 'success'
    for idx in result['out'].index:
        assert result['out'].loc[idx, "success"] == pytest.approx(
            test_out.loc[idx, "success"], 0.1)


def test_quantile_discretizer_uniform_output_distribution_param_success():
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
    instance = QDO(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    model_1 = KBinsDiscretizer(n_bins=1000,
                               encode='ordinal', strategy='uniform')
    X_train = get_X_train_data(test_df, ['sepallength'])
    test_df['quantiledisc_1'] = model_1.fit_transform(X_train).flatten().tolist()

    assert result['out'].equals(test_df)


def test_quantile_discretizer_kmeans_output_distribution_param_success():
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
    instance = QDO(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    model_1 = KBinsDiscretizer(n_bins=2,
                               encode='ordinal', strategy='kmeans')
    X_train = get_X_train_data(test_df, ['sepallength'])
    test_df['quantiledisc_1'] = model_1.fit_transform(X_train).flatten().tolist()

    assert result['out'].equals(test_df)


def test_quantile_discretizer_quantile_output_distribution_param_success():
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
    instance = QDO(**arguments)
    result = util.execute(util.get_complete_code(instance), {'df': df})

    model_1 = KBinsDiscretizer(n_bins=1000,
                               encode='ordinal', strategy='quantile')
    X_train = get_X_train_data(test_df, ['sepallength'])
    test_df['quantiledisc_1'] = model_1.fit_transform(X_train).flatten().tolist()

    assert result['out'].equals(test_df)


def test_quantile_discretizer_no_output_implies_no_code_success():
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
    instance = QDO(**arguments)
    assert instance.generate_code() is None


def test_quantile_discretizer_missing_input_implies_no_code_success():
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
    instance = QDO(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_quantile_discretizer_invalid_n_quantiles_param_fail():
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
        QDO(**arguments)
    assert "Parameter 'n_quantiles' must be x>0 for task" in str(val_err.value)


def test_quantile_discretizer_missing_attribute_param_fail():
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
        QDO(**arguments)
    assert "Parameters 'attributes' must be informed for task" in str(
        val_err.value)
