from tests.scikit_learn import util
from tests.scikit_learn.util import get_X_train_data, get_label_data
from juicer.scikit_learn.classification_operation import \
    NaiveBayesClassifierModelOperation, NaiveBayesClassifierOperation
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
import pytest
import pandas as pd
import numpy as np


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)


# NaiveBayesClassifier
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_naive_bayes_classifier_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MultinomialNB(alpha=1.0,
                            class_prior=None, fit_prior=True)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_naive_bayes_classifier_alpha_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MultinomialNB(alpha=2.0,
                            class_prior=None, fit_prior=True)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_naive_bayes_classifier_class_prior_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
    test_df = df.copy()
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'train input data': 0},
                       'label': ['sepalwidth'],
                       'class_prior': "0.1, 0.1, 0.1, 0.1, 0.1,"
                                      " 0.1, 0.1, 0.1, 0.1, 0.1",
                       },
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
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MultinomialNB(alpha=1.0,
                            class_prior=[0.1, 0.1, 0.1, 0.1, 0.1,
                                         0.1, 0.1, 0.1, 0.1, 0.1],
                            fit_prior=True)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_naive_bayes_classifier_fit_prior_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=False)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_naive_bayes_classifier_type_gaussiannb_var_smoothing_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = GaussianNB(priors=None,
                         var_smoothing=2.0)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


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
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = GaussianNB(priors=[0.5, 0.5],
                         var_smoothing=1e-09)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_naive_bayes_classifier_type_bernoulli_binarize_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    x_train = get_X_train_data(test_df, ['sepallength', 'sepalwidth'])
    y = get_label_data(test_df, ['sepalwidth'])
    y = np.reshape(y, len(y))
    model_1 = BernoulliNB(alpha=1.0,
                          class_prior=None, fit_prior=True,
                          binarize=10.0)
    model_1.fit(x_train, y)
    test_df['prediction'] = model_1.predict(x_train).tolist()
    assert result['out'].equals(test_df)
    assert str(result['model_task_1']) == str(model_1)


def test_naive_bayes_classifier_prediction_param_success():
    df = pd.DataFrame(
        [[0, 1], [1, 2],
         [2, 3], [3, 4],
         [4, 5], [5, 6],
         [6, 7], [7, 8],
         [8, 9], [9, 10]], columns=['sepallength', 'sepalwidth'])
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
    assert f"Parameter 'alpha' must be x>0 for task" \
           f" {NaiveBayesClassifierOperation}" in str(val_err.value)
