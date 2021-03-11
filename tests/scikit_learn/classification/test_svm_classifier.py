from tests.scikit_learn import util
from juicer.scikit_learn.classification_operation \
    import SvmClassifierModelOperation
from sklearn.svm import SVC
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# SvmClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_svm_classifier_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_c_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = SVC(tol=0.001, C=2.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_kernel_param_success_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='linear', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_degree_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=2, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_tol_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = SVC(tol=0.1, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_max_iter_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = SVC(tol=0.001, C=1.0, max_iter=2,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_seed_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=2002,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_prediction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[2] == 'success'


def test_svm_classifier_gamma_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='auto', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_coef0_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=1.0, probability=False,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_shrinking_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=False, decision_function_shape='ovr',
                  class_weight=None)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_probability_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=True,
                  shrinking=True, decision_function_shape='ovr',
                  class_weight=None)
    assert not result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_svm_classifier_decision_function_shape_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = SvmClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = SVC(tol=0.001, C=1.0, max_iter=-1,
                  degree=3, kernel='rbf', random_state=None,
                  gamma='scale', coef0=0.0, probability=False,
                  shrinking=True, decision_function_shape='ovo',
                  class_weight=None)
    assert not result['out'].equals(test_df)
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
    arguments = util.add_minimum_ml_args(arguments)
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
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        SvmClassifierModelOperation(**arguments)
    assert "Parameters 'features' and 'label' must be informed for task " \
           in str(val_err.value)


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
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        SvmClassifierModelOperation(**arguments)
    assert "Parameters 'features' and 'label' must be informed for task " in \
           str(val_err.value)


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
        arguments = util.add_minimum_ml_args(arguments)
        with pytest.raises(ValueError) as val_err:
            SvmClassifierModelOperation(**arguments)
        assert f"Parameter '{par}' must be x>0 for task" in str(val_err.value)
