from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import StandardScalerOperation
from sklearn.preprocessing import StandardScaler
from tests.scikit_learn.util import get_X_train_data
from textwrap import dedent
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# StandardScaler
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_standard_scaler_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = StandardScalerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = StandardScaler(with_mean=False, with_std=True)
    values = model_1.fit_transform(X_train)

    test_df = pd.concat(
        [test_df,
         pd.DataFrame(values,
                      columns=['sepallength_norm',
                               'sepalwidth_norm'])], ignore_index=False, axis=1)

    assert result['out'].equals(test_df)
    assert str(model_1) == str(result['model_1'])
    assert dedent("""
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])

    model_1 = StandardScaler(with_mean=False, with_std=True)
    values = model_1.fit_transform(X_train)

    out = pd.concat([df, 
        pd.DataFrame(values, columns=['sepallength_norm', 'sepalwidth_norm'])],
        ignore_index=False, axis=1)
    """) == instance.generate_code()


def test_standard_scaler_alias_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'alias': 'success1,success2'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = StandardScalerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = StandardScaler(with_mean=False, with_std=True)
    values = model_1.fit_transform(X_train)

    test_df = pd.concat(
        [test_df,
         pd.DataFrame(values,
                      columns=['success1',
                               'success2'])], ignore_index=False, axis=1)

    assert result['out'].equals(test_df)
    assert str(model_1) == str(result['model_1'])
    assert dedent("""
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])

    model_1 = StandardScaler(with_mean=False, with_std=True)
    values = model_1.fit_transform(X_train)

    out = pd.concat([df, 
        pd.DataFrame(values, columns=['success1', 'success2'])],
        ignore_index=False, axis=1)
    """) == instance.generate_code()


def test_standard_scaler_with_mean_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'with_mean': 1},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = StandardScalerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = StandardScaler(with_mean=True)
    values = model_1.fit_transform(X_train)

    test_df = pd.concat(
        [test_df,
         pd.DataFrame(values,
                      columns=['sepallength_norm',
                               'sepalwidth_norm'])], ignore_index=False, axis=1)

    assert result['out'].equals(test_df)
    assert str(model_1) == str(result['model_1'])
    assert dedent("""
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])

    model_1 = StandardScaler(with_mean=True, with_std=True)
    values = model_1.fit_transform(X_train)

    out = pd.concat([df, 
        pd.DataFrame(values, columns=['sepallength_norm', 'sepalwidth_norm'])],
        ignore_index=False, axis=1)
    """) == instance.generate_code()


def test_standard_scaler_with_std_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'with_std': 0, 'with_mean': 0},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = StandardScalerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    model_1 = StandardScaler(with_std=False, with_mean=False)
    values = model_1.fit_transform(X_train)

    test_df = pd.concat(
        [test_df,
         pd.DataFrame(values,
                      columns=['sepallength_norm',
                               'sepalwidth_norm'])], ignore_index=False, axis=1)

    assert result['out'].equals(test_df)
    assert str(model_1) == str(result['model_1'])
    assert dedent("""
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])

    model_1 = StandardScaler(with_mean=False, with_std=False)
    values = model_1.fit_transform(X_train)

    out = pd.concat([df, 
        pd.DataFrame(values, columns=['sepallength_norm', 'sepalwidth_norm'])],
        ignore_index=False, axis=1)
    """) == instance.generate_code()


def test_standard_scaler_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = StandardScalerOperation(**arguments)
    assert instance.generate_code() is None


def test_standard_scaler_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = StandardScalerOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_standard_missing_attribute_param_scaler_fail():
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
        StandardScalerOperation(**arguments)
    assert f"Parameters 'attributes' must be informed for task" \
           f" StandardScalerOperation" in str(val_err.value)
