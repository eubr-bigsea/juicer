from tests.scikit_learn import util
from juicer.scikit_learn.classification_operation import \
    PerceptronClassifierOperation
from sklearn.linear_model import Perceptron
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# PerceptronClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_perceptron_classifier_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth']},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.001, alpha=0.0001, max_iter=1000, shuffle=False,
                         random_state=None, penalty='None', fit_intercept=1,
                         eta0=1.0, n_jobs=None, early_stopping=False,
                         n_iter_no_change=5, class_weight=None,
                         warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_alpha_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'alpha': 0.1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.001, alpha=0.1, max_iter=1000, shuffle=False,
                         random_state=None, penalty='None', fit_intercept=1,
                         eta0=1.0, n_jobs=None, early_stopping=False,
                         n_iter_no_change=5, class_weight=None,
                         warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_tol_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'tol': 0.1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.1, alpha=0.0001, max_iter=1000, shuffle=False,
                         random_state=None, penalty='None', fit_intercept=1,
                         eta0=1.0, n_jobs=None, early_stopping=False,
                         n_iter_no_change=5, class_weight=None,
                         warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_shuffle_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'shuffle': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.001, alpha=0.0001, max_iter=1000, shuffle=True,
                         random_state=None, penalty='None', fit_intercept=1,
                         eta0=1.0, n_jobs=None, early_stopping=False,
                         n_iter_no_change=5, class_weight=None,
                         warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_seed_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'seed': 2002},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.001, alpha=0.0001, max_iter=1000, shuffle=False,
                         random_state=2002, penalty='None', fit_intercept=1,
                         eta0=1.0, n_jobs=None, early_stopping=False,
                         n_iter_no_change=5, class_weight=None,
                         warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_penalty_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'penalty': 'l1'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.001, alpha=0.0001, max_iter=1000, shuffle=False,
                         random_state=None, penalty='l1', fit_intercept=1,
                         eta0=1.0, n_jobs=None, early_stopping=False,
                         n_iter_no_change=5, class_weight=None,
                         warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_max_iter_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'max_iter': 500},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.001, alpha=0.0001, max_iter=500, shuffle=False,
                         random_state=None, penalty='None', fit_intercept=1,
                         eta0=1.0, n_jobs=None, early_stopping=False,
                         n_iter_no_change=5, class_weight=None,
                         warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_fit_intercept_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'fit_intercept': 2},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.001, alpha=0.0001, max_iter=1000, shuffle=False,
                         random_state=None, penalty='None', fit_intercept=2,
                         eta0=1.0, n_jobs=None, early_stopping=False,
                         n_iter_no_change=5, class_weight=None,
                         warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_eta0_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'eta0': 2.0},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.001, alpha=0.0001, max_iter=1000, shuffle=False,
                         random_state=None, penalty='None', fit_intercept=1,
                         eta0=2.0, n_jobs=None, early_stopping=False,
                         n_iter_no_change=5, class_weight=None,
                         warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_n_jobs_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'n_jobs': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.001, alpha=0.0001, max_iter=1000, shuffle=False,
                         random_state=None, penalty='None', fit_intercept=1,
                         eta0=1.0, n_jobs=1, early_stopping=False,
                         n_iter_no_change=5, class_weight=None,
                         warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_early_stopping_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=150)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'early_stopping': 1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.001, alpha=0.0001, max_iter=1000, shuffle=False,
                         random_state=None, penalty='None', fit_intercept=1,
                         eta0=1.0, n_jobs=None, early_stopping=True,
                         n_iter_no_change=5, class_weight=None,
                         warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_validation_fraction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=150)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'early_stopping': 1,
                       'validation_fraction': 0.5},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.001, alpha=0.0001, max_iter=1000, shuffle=False,
                         random_state=None, penalty='None', fit_intercept=1,
                         eta0=1.0, n_jobs=None, early_stopping=True,
                         validation_fraction=0.5, n_iter_no_change=5,
                         class_weight=None, warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_n_iter_no_change_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'n_iter_no_change': 10},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.001, alpha=0.0001, max_iter=1000, shuffle=False,
                         random_state=None, penalty='None', fit_intercept=1,
                         eta0=1.0, n_jobs=None, early_stopping=False,
                         n_iter_no_change=10, class_weight=None,
                         warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_class_weight_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'class_weight': "'balanced'"},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = Perceptron(tol=0.001, alpha=0.0001, max_iter=1000, shuffle=False,
                         random_state=None, penalty='None', fit_intercept=1,
                         eta0=1.0, n_jobs=None, early_stopping=False,
                         n_iter_no_change=5, class_weight='balanced',
                         warm_start=False)
    assert not result['out'].equals(test_df)
    assert str(result['model_1']) == str(model_1)


def test_perceptron_classifier_prediction_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'prediction': 'success'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[2] == 'success'


def test_perceptron_classifier_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth']},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    assert instance.generate_code() is None


def test_perceptron_classifier_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth']},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = PerceptronClassifierOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_perceptron_classifier_invalid_max_iter_and_alpha_params_fail():
    vals = [
        'max_iter',
        'alpha'
    ]

    for par in vals:
        arguments = {
            'parameters': {'features': ['sepallength', 'sepalwidth'],
                           'multiplicity': {'train input data': 0},
                           'label': ['sepalwidth'],
                           par: -1},
            'named_inputs': {
                'train input data': 'df',
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        with pytest.raises(ValueError) as val_err:
            PerceptronClassifierOperation(**arguments)
        assert f"Parameter '{par}' must be x>0 for task" in str(val_err.value)


def test_perceptron_classifier_invalid_validation_fraction_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'validation_fraction': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        PerceptronClassifierOperation(**arguments)
    assert "Parameter 'validation_fraction' must be 0 <= x =< 1 for task" in \
           str(val_err.value)
