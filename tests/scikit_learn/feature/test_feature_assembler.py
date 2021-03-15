from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import FeatureAssemblerOperation
from textwrap import dedent
import pytest
import pandas as pd
import numpy as np


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# FeatureAssembler
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_feature_assembler_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FeatureAssemblerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    cols = ['sepalwidth', 'petalwidth']
    test_df = test_df.dropna(subset=cols)
    test_df['FeatureField'] = test_df[cols].to_numpy().tolist()
    assert result['out'].equals(test_df)
    assert dedent("""
    cols = ['sepalwidth', 'petalwidth']
    if df[cols].dtypes.all() == np.object:
        raise ValueError("Input 'df' must contain numeric values"
        " only for task <class 'juicer.scikit_learn.feature_operation.FeatureAssemblerOperation'>")
           
    out = df.dropna(subset=cols)
    out['FeatureField'] = out[cols].to_numpy().tolist()
    """) == instance.generate_code()


def test_feature_assembler_alias_param_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_df = df.copy()

    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0},
                       'alias': 'Feat'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FeatureAssemblerOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'df': df})

    cols = ['sepalwidth', 'petalwidth']
    test_df = test_df.dropna(subset=cols)
    test_df['Feat'] = test_df[cols].to_numpy().tolist()
    assert result['out'].equals(test_df)
    assert dedent("""
    cols = ['sepalwidth', 'petalwidth']
    if df[cols].dtypes.all() == np.object:
        raise ValueError("Input 'df' must contain numeric values"
        " only for task <class 'juicer.scikit_learn.feature_operation.FeatureAssemblerOperation'>")

    out = df.dropna(subset=cols)
    out['Feat'] = out[cols].to_numpy().tolist()
    """) == instance.generate_code()


def test_feature_assembler_na_drop_success():
    """
    The chained assignment warnings / exceptions are aiming to inform the user
    of a possibly invalid assignment. There is a false positives; situations
    where a chained assignment is inadvertently reported.
    https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    In this case, the error is due to util.execute()
    """
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)

    df.loc[0:2, 'sepalwidth'] = np.NaN

    test_df = df.copy()
    cols = ['sepalwidth', 'petalwidth']
    test_df = test_df.dropna(subset=cols)
    test_df['FeatureField'] = test_df[cols].to_numpy().tolist()

    arguments = {
        'parameters': {'attributes': cols,
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FeatureAssemblerOperation(**arguments)
    result = util.execute(instance.generate_code(), {'df': df})
    assert result['out'].equals(test_df)
    assert dedent("""
    cols = ['sepalwidth', 'petalwidth']
    if df[cols].dtypes.all() == np.object:
        raise ValueError("Input 'df' must contain numeric values"
        " only for task <class 'juicer.scikit_learn.feature_operation.FeatureAssemblerOperation'>")

    out = df.dropna(subset=cols)
    out['FeatureField'] = out[cols].to_numpy().tolist()
    """) == instance.generate_code()


def test_feature_assembler_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = FeatureAssemblerOperation(**arguments)
    assert instance.generate_code() is None


def test_feature_assembler_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FeatureAssemblerOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_feature_assembler_missing_attributes_param_fail():
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
        FeatureAssemblerOperation(**arguments)
    assert f"Parameters 'attributes' must be informed for task" \
           f" {FeatureAssemblerOperation}" in str(val_err.value)


def test_feature_assembler_invalid_dtype_input_fail():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    test_out = {
        'Feat': [[3.1, 0.2], [3.5, 0.3], [3.3, 0.4], [3.1, 0.2],
                 [3.6, 0.2], [3.9, 0.4], [3.4, 0.3], [3.4, 0.2],
                 [2.9, 0.2], [3.1, 0.1]]}
    test_out_2 = {
        'Feat2': ['3.1', '0.3', '0.4', '0.2',
                  '3.6', '3.9', '0.3', '0.2',
                  '2.9', '0.1']}
    test_out = pd.DataFrame(test_out)
    test_out_2 = pd.DataFrame(test_out_2)
    df = pd.concat([df, test_out, test_out_2], axis=1, join='inner')
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'Feat', 'Feat2'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FeatureAssemblerOperation(**arguments)
    with pytest.raises(ValueError) as val_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert f"Input 'df' must contain numeric values only for task" \
           f" {FeatureAssemblerOperation}" in str(val_err.value)
