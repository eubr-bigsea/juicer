
from tests.scikit_learn import util
from juicer.scikit_learn.classification_operation import \
    NaiveBayesClassifierModelOperation
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# NaiveBayesClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_naive_bayes_classifier_success():
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = MultinomialNB(alpha=1.0,
                            class_prior=None, fit_prior=True)
    assert str(model_1) == str(result['model_task_1'])
    assert not result['out'].equals(test_df)


def test_naive_bayes_classifier_alpha_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'alpha': 2.0},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = MultinomialNB(alpha=2.0,
                            class_prior=None, fit_prior=True)
    assert str(model_1) == str(result['model_task_1'])
    assert not result['out'].equals(test_df)


# Todo
def xtest_naive_bayes_classifier_class_prior_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'class_prior': '1'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = MultinomialNB(alpha=1.0,
                            class_prior=None, fit_prior=True)
    assert str(model_1) == str(result['model_task_1'])
    assert not result['out'].equals(test_df)


def test_naive_bayes_classifier_fit_prior_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'fit_prior': 0},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=False)
    assert str(model_1) == str(result['model_task_1'])
    assert not result['out'].equals(test_df)


def test_naive_bayes_classifier_type_gaussiannb_var_smoothing_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'var_smoothing': 2,
                       'type': 'GaussianNB'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = GaussianNB(priors=None,
                         var_smoothing=2.0)
    assert str(model_1) == str(result['model_task_1'])
    assert not result['out'].equals(test_df)


def test_naive_bayes_classifier_type_gaussiannb_priors_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'priors': "0.5, 0.5",
                       'type': 'GaussianNB'},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})

    model_1 = GaussianNB(priors=[0.5, 0.5],
                         var_smoothing=1e-09)
    assert str(model_1) == str(result['model_task_1'])
    assert not result['out'].equals(test_df)


def test_naive_bayes_classifier_type_bernoulli_binarize_param_success():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    for idx in df.index:
        for col in df.columns:
            df.loc[idx, col] = int(df.loc[idx, col])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'type': 'Bernoulli',
                       'binarize': 10},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    model_1 = BernoulliNB(alpha=1.0,
                          class_prior=None, fit_prior=True,
                          binarize=10.0)
    assert str(model_1) == str(result['model_task_1'])
    assert not result['out'].equals(test_df)


def test_naive_bayes_classifier_prediction_param_success():
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df})
    assert result['out'].columns[2] == 'success'


def test_naive_bayes_classifier_no_output_implies_no_code_success():
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


def test_naive_bayes_classifier_missing_input_implies_no_code_success():
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
    arguments = util.add_minimum_ml_args(arguments)
    instance = NaiveBayesClassifierModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_naive_bayes_classifier_invalid_alpha_param_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'type': 'Multinomial',
                       'alpha': -1},
        'named_inputs': {
            'train input data': 'df',
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    arguments = util.add_minimum_ml_args(arguments)
    with pytest.raises(ValueError) as val_err:
        NaiveBayesClassifierModelOperation(**arguments)
    assert "Parameter 'alpha' must be x>0 for task" in str(val_err.value)
