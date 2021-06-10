from tests.scikit_learn import util
from tests.scikit_learn.util import get_label_data, get_X_train_data
from juicer.scikit_learn.classification_operation \
    import SvmClassifierModelOperation, SvmClassifierOperation
from sklearn.svm import SVC
import pytest
import pandas as pd
import numpy as np


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# SvmClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_svm_classifier_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_c_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'c': 2.0},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = SVC(tol=0.001, C=2.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_kernel_param_success_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'kernel': 'linear'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='linear', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_degree_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'degree': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=2, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_tol_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'tol': 0.1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = SVC(tol=0.1, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_max_iter_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'max_iter': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = SVC(tol=0.001, C=1.0, max_iter=2,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_seed_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'seed': 2002},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=2002,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_prediction_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'prediction': 'success'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[2] == 'success'


def test_svm_classifier_gamma_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'gamma': 'auto'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='auto', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_coef0_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'coef0': 1.0},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=1.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_shrinking_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'shrinking': 0},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=False, decision_function_shape='ovr',
                  class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_probability_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'probability': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=True,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_decision_function_shape_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]],
        columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'decision_function_shape': 'ovo'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    X_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovo',
                  class_weight=None)
    model_1.fit(X_train, y)
    test_df['prediction'] = model_1.predict(X_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
        }
    }
    util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


def test_svm_classifier_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_svm_classifier_missing_label_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        SvmClassifierModelOperation(**arguments)
    assert "Parameters 'features' and 'label' must be informed for task" \
           " ClassificationModelOperation" in str(val_err.value)


def test_svm_classifier_missing_features_param_fail():
    arguments = {
        'parameters': {'label': ['sepalwidth'],
                       'multiplicity': {'train input data': 0}},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        SvmClassifierModelOperation(**arguments)
    assert "Parameters 'features' and 'label' must be informed for task" \
           " ClassificationModelOperation" in str(val_err.value)


def test_svm_classifier_multiple_invalid_params_fail():
    vals = [
        'degree',
        'c'
    ]
    for par in vals:
        arguments = {
            'parameters': {'features': ['sepallength', 'sepalwidth'],
                           'label': ['sepalwidth'],
                           'multiplicity': {'train input data': 0},
                           par: -1},
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        util.add_minimum_ml_args(arguments)
        with pytest.raises(ValueError) as val_err:
            SvmClassifierModelOperation(**arguments)
        assert f"Parameter '{par}' must be x>0 for task" \
               f" {SvmClassifierOperation}" in str(val_err.value)
