from tests.scikit_learn import util
from juicer.scikit_learn.feature_operation import StringIndexerOperation
from tests.scikit_learn.util import get_X_train_data
from textwrap import dedent
from sklearn.preprocessing import LabelEncoder
import pytest


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# StringIndexer
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_string_indexer_success():
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
    instance = StringIndexerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    le = LabelEncoder()
    for col, new_col in zip(['sepallength', 'sepalwidth'],
                            ['sepallength_indexed', 'sepalwidth_indexed']):
        data = test_df[col].to_numpy().tolist()
        test_df[new_col] = le.fit_transform(data)

    assert result['out'].equals(test_df)
    assert str(result['le']) == str(le)
    assert dedent("""
    out = df
    models_task_1 = dict()
    le = LabelEncoder()
    for col, new_col in zip(['sepallength', 'sepalwidth'], ['sepallength_indexed', 'sepalwidth_indexed']):
        data = df[col].to_numpy().tolist()
        models_task_1[new_col] = le.fit_transform(data)
        out[new_col] =le.fit_transform(data)    
    """) == instance.generate_code()


def test_string_indexer_alias_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    test_df = df.copy()
    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'alias': 'success_1,success_2'},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = StringIndexerOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    le = LabelEncoder()
    for col, new_col in zip(['sepallength', 'sepalwidth'],
                            ['success_1', 'success_2']):
        data = test_df[col].to_numpy().tolist()
        test_df[new_col] = le.fit_transform(data)
    assert result['out'].equals(test_df)
    assert str(result['le']) == str(le)
    assert dedent("""
    out = df
    models_task_1 = dict()
    le = LabelEncoder()
    for col, new_col in zip(['sepallength', 'sepalwidth'], ['success_1', 'success_2']):
        data = df[col].to_numpy().tolist()
        models_task_1[new_col] = le.fit_transform(data)
        out[new_col] =le.fit_transform(data)    
    """) == instance.generate_code()


def test_string_indexer_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = StringIndexerOperation(**arguments)
    assert instance.generate_code() is None


def test_string_indexer_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'attributes': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = StringIndexerOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_string_indexer_missing_attributes_param_fail():
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
        StringIndexerOperation(**arguments)
    assert f"Parameter 'attributes' must be informed for task " \
           f"{StringIndexerOperation}" in str(val_err.value)
