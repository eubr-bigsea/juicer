from tests.scikit_learn import util
from juicer.scikit_learn.classification_operation import \
    ClassificationModelOperation

import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# ClassificationModel
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_classification_model_svc_algorithm_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
            'algorithm': 'SVC()'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ClassificationModelOperation(**arguments)
    instance.transpiler_utils.add_import('from sklearn.svm import SVC')
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert not result['out'].equals(test_df)


def test_classification_model_randomforestclassifier_algorithm_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
            'algorithm': 'RandomForestClassifier()'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ClassificationModelOperation(**arguments)
    instance.transpiler_utils.add_import(
        'from sklearn.ensemble import RandomForestClassifier')
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert not result['out'].equals(test_df)


def test_classification_model_missing_output_impleis_no_code_success():
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
            'algorithm': 'SVC()'
        },
        'named_outputs': {
        }
    }
    instance = ClassificationModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_classification_model_missing_features_and_label_params_fail():
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
            'algorithm': 'SVC()'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    arguments_2 = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
            'algorithm': 'SVC()'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }

    with pytest.raises(ValueError) as val_err:
        ClassificationModelOperation(**arguments)
    assert "Parameters 'features' and 'label' must be informed for task" in str(
        val_err.value)

    with pytest.raises(ValueError) as val_err_2:
        ClassificationModelOperation(**arguments_2)
    assert "Parameters 'features' and 'label' must be informed for task" in str(
        val_err_2.value)


def test_classification_model_missing_one_input_fail():
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'features': ['sepallength', 'sepalwidth']},
        'named_inputs': {
            'algorithm': 'SVC()'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        ClassificationModelOperation(**arguments)
    assert "Model is being used, but at least one input is missing" in \
           str(val_err.value)
