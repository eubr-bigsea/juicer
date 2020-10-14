from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import FeatureAssemblerOperation
import pytest
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


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

    test_out = {'FeatureField': [[3.5, 0.2], [3.0, 0.2], [3.2, 0.2], [3.1, 0.2],
                                 [3.6, 0.2], [3.9, 0.4], [3.4, 0.3], [3.4, 0.2],
                                 [2.9, 0.2], [3.1, 0.1]]}
    test_out = pd.DataFrame(test_out)
    test_out = pd.concat([test_df, test_out], axis=1)
    assert result['out'].equals(test_out)


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

    test_out = {'Feat': [[3.5, 0.2], [3.0, 0.2], [3.2, 0.2], [3.1, 0.2],
                         [3.6, 0.2], [3.9, 0.4], [3.4, 0.3], [3.4, 0.2],
                         [2.9, 0.2], [3.1, 0.1]]}
    test_out = pd.DataFrame(test_out)
    test_out = pd.concat([test_df, test_out], axis=1)
    assert result['out'].equals(test_out)


def test_feature_assembler_na_drop_success():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    df.iloc[0:3, 1] = np.NaN
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

    test_out = {
        'FeatureField': [np.NaN, np.NaN, np.NaN, [3.1, 0.2], [3.6, 0.2],
                         [3.9, 0.4], [3.4, 0.3], [3.4, 0.2], [2.9, 0.2],
                         [3.1, 0.1]]}
    test_out = pd.DataFrame(test_out)
    test_out = pd.concat([test_df, test_out], axis=1)
    test_out.drop([0, 1, 2], axis=0, inplace=True)
    assert result['out'].equals(test_out)


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
    assert "Parameters 'attributes' must be informed for task" in str(
        val_err.value)


def test_feature_assembler_missing_multiplicity_param_fail():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth']},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FeatureAssemblerOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "'multiplicity'" in str(key_err.value)


def test_feature_assembler_invalid_attributes_param_fail():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    arguments = {
        'parameters': {'attributes': 'invalid',
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FeatureAssemblerOperation(**arguments)
    with pytest.raises(NameError) as nam_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "name 'invalid' is not defined" in str(nam_err)


def test_feature_assembler_invalid_multiplicity_param_fail():
    df = util.iris(['sepallength', 'sepalwidth',
                    'petalwidth', 'petallength'], size=10)
    arguments = {
        'parameters': {'attributes': ['sepalwidth', 'petalwidth'],
                       'multiplicity': 'invalid'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = FeatureAssemblerOperation(**arguments)
    with pytest.raises(TypeError) as typ_err:
        util.execute(instance.generate_code(),
                     {'df': df})
    assert "string indices must be integers" in str(typ_err)
