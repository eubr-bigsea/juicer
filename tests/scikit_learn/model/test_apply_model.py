from tests.scikit_learn import util
from juicer.scikit_learn.model_operation import ApplyModelOperation
from juicer.scikit_learn.util import get_X_train_data, get_label_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# ApplyModel
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_apply_model_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=50)

    test_df = df.copy()
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    model = MinMaxScaler().fit(X_train)
    test_df['prediction'] = model.transform(X_train).tolist()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
            'model': 'model'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ApplyModelOperation(**arguments)
    instance.transpiler_utils.add_import(
        "from sklearn.preprocessing import MinMaxScaler")
    result = util.execute(util.get_complete_code(instance),
                          {'df': df, 'model': model})

    assert result['out'].equals(test_df)
    assert instance.generate_code() == """
out = df
X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
if hasattr(model, 'predict'):
    out['prediction'] = model.predict(X_train).tolist()
else:
    # to handle scaler operations
    out['prediction'] = model.transform(X_train).tolist()
"""


def test_apply_model_success_2():
    df = util.iris(['sepallength', 'sepalwidth', 'class'], size=10)

    test_df = df.copy()
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    y = get_label_data(df, ['class'])
    knn_model = KNeighborsClassifier().fit(X_train, y)
    test_df['prediction'] = knn_model.predict(X_train).tolist()

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
            'model': 'knn_model'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ApplyModelOperation(**arguments)
    instance.transpiler_utils.add_import(
        "from sklearn.neighbors import KNeighborsClassifier")
    result = util.execute(util.get_complete_code(instance),
                          {'df': df, 'knn_model': knn_model})
    assert result['out'].equals(test_df)
    assert instance.generate_code() == """
out = df
X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
if hasattr(knn_model, 'predict'):
    out['prediction'] = knn_model.predict(X_train).tolist()
else:
    # to handle scaler operations
    out['prediction'] = knn_model.transform(X_train).tolist()
"""


def test_apply_model_prediction_param_success():
    df = util.iris(['sepallength', 'sepalwidth', 'class'], size=10)
    ncols = len(df.columns)
    X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
    y = get_label_data(df, ['class'])
    knn_model = KNeighborsClassifier().fit(X_train, y)

    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'prediction': 'success'
                       },
        'named_inputs': {
            'input data': 'df',
            'model': 'knn_model'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = ApplyModelOperation(**arguments)
    instance.transpiler_utils.add_import(
            "from sklearn.neighbors import KNeighborsClassifier")
    result = util.execute(util.get_complete_code(instance),
                          {'df': df, 'knn_model': knn_model})
    assert ncols + 1 == len(result['out'].columns)
    predicted_labels = list(set(result['out']['success'].to_numpy().tolist()))
    labels = set(y)
    assert all([p in labels for p in predicted_labels])


def test_apply_model_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'prediction': 'success'},
        'named_inputs': {
            'input data': 'df',
            'model': MinMaxScaler()
        },
        'named_outputs': {
        }
    }
    instance = ApplyModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # # Fail # # # # # # # # # #
def test_apply_model_missing_features_param_fail():
    arguments = {
        'parameters': {'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
            'model': MinMaxScaler()
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        ApplyModelOperation(**arguments)
    assert "Parameters 'features' must be informed for task" in str(
        val_err.value)


def test_apply_model_missing_one_input_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'model': MinMaxScaler()
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        ApplyModelOperation(**arguments)
    assert "Model is being used, but at least one input is missing" in str(
        val_err.value)
