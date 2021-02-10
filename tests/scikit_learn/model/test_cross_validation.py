from tests.scikit_learn import util
from juicer.scikit_learn.model_operation import CrossValidationOperation
from sklearn.neural_network import MLPClassifier
import pytest
import pandas as pd


# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# CrossValidation
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_cross_validation_success():
    data = {'sepallength': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'sepalwidth': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
    df = pd.DataFrame(data)
    test_df = df.copy()
    arguments = {
        'parameters': {'evaluator': 'accuracy',
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'label_attribute': ['sepalwidth']},
        'named_inputs': {
            'input data': 'df',
            'algo_1': MLPClassifier()
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CrossValidationOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df,
                           'algo_1': MLPClassifier()})
    assert not result['scored_data_task_1'].equals(test_df)
    assert instance.generate_code() == """
kf = KFold(n_splits=3, random_state=None, 
shuffle=True)
X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
X_train = np.array(X_train)
y = get_label_data(df, ['sepalwidth'])
scores = cross_val_score(algo_1, X_train, 
                         y, cv=kf, scoring='accuracy')

best_score = np.argmax(scores)

models = None
train_index, test_index = list(kf.split(X_train))[best_score]
Xf_train, Xf_test = X_train[train_index], X_train[test_index]
yf_train, yf_test = y[train_index],  y[test_index]
best_model_1 = algo_1.fit(Xf_train.tolist(), yf_train.tolist())

metric_result = scores[best_score]
scored_data_task_1 = df
scored_data_task_1['prediction'] = best_model_1.predict(X_train.tolist())
models_task_1 = models
"""


def test_cross_validation_prediction_attribute_param_success():
    data = {'sepallength': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'sepalwidth': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
    df = pd.DataFrame(data)
    arguments = {
        'parameters': {'evaluator': 'accuracy',
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'label_attribute': ['sepalwidth'],
                       'prediction_attribute': 'success'},
        'named_inputs': {
            'input data': 'df',
            'algo_1': MLPClassifier()
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CrossValidationOperation(**arguments)
    result = util.execute(util.get_complete_code(instance),
                          {'df': df,
                           'algo_1': MLPClassifier()})
    assert result['scored_data_task_1'].columns[2] == 'success'


def test_cross_validation_folds_and_seed_params_success():
    data = {'sepallength': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'sepalwidth': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
    df = pd.DataFrame(data)
    arguments = {
        'parameters': {'evaluator': 'accuracy',
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'label_attribute': ['sepalwidth'],
                       'folds': 5,
                       'seed': 2002},
        'named_inputs': {
            'input data': 'df',
            'algo_1': MLPClassifier()
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CrossValidationOperation(**arguments)
    assert instance.generate_code() == """
kf = KFold(n_splits=5, random_state=2002, 
shuffle=True)
X_train = get_X_train_data(df, ['sepallength', 'sepalwidth'])
X_train = np.array(X_train)
y = get_label_data(df, ['sepalwidth'])
scores = cross_val_score(algo_1, X_train, 
                         y, cv=kf, scoring='accuracy')

best_score = np.argmax(scores)

models = None
train_index, test_index = list(kf.split(X_train))[best_score]
Xf_train, Xf_test = X_train[train_index], X_train[test_index]
yf_train, yf_test = y[train_index],  y[test_index]
best_model_1 = algo_1.fit(Xf_train.tolist(), yf_train.tolist())

metric_result = scores[best_score]
scored_data_task_1 = df
scored_data_task_1['prediction'] = best_model_1.predict(X_train.tolist())
models_task_1 = models
"""


def test_cross_validation_no_output_implies_no_code_success():
    arguments = {
        'parameters': {'evaluator': 'mse',
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'label_attribute': ['sepalwidth']},
        'named_inputs': {
            'input data': 'df',
            'algo_1': MLPClassifier()
        },
        'named_outputs': {
        }
    }
    instance = CrossValidationOperation(**arguments)
    assert instance.generate_code() is None


def test_cross_validation_missing_input_implies_no_code_success():
    arguments = {
        'parameters': {'evaluator': 'mse',
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'label_attribute': ['sepalwidth']},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = CrossValidationOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_cross_validation_missing_evaluator_param_fail():
    arguments = {
        'parameters': {'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'label_attribute': ['sepalwidth']},
        'named_inputs': {
            'input data': 'df',
            'algo_1': MLPClassifier()
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        CrossValidationOperation(**arguments)
    assert "Parameter 'evaluator' must be informed for task" in str(val_err)


def test_cross_validation_invalid_metric_fail():
    arguments = {
        'parameters': {'evaluator': 'invalid',
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0},
                       'label_attribute': ['sepalwidth']},
        'named_inputs': {
            'input data': 'df',
            'algo_1': MLPClassifier()
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        CrossValidationOperation(**arguments)
    assert "Invalid metric value invalid" in str(val_err.value)


def test_cross_validation_missing_features_param_fail():
    arguments = {
        'parameters': {'evaluator': 'accuracy',
                       'multiplicity': {'input data': 0},
                       'label_attribute': ['sepalwidth']},
        'named_inputs': {
            'input data': 'df',
            'algo_1': MLPClassifier()
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        CrossValidationOperation(**arguments)
    assert "Parameter 'label_attribute' must be informed for task" in str(
        val_err.value)


def test_cross_validation_missing_label_param_fail():
    arguments = {
        'parameters': {'evaluator': 'accuracy',
                       'features': ['sepallength', 'sepalwidth'],
                       'multiplicity': {'input data': 0}},
        'named_inputs': {
            'input data': 'df',
            'algo_1': MLPClassifier()
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        CrossValidationOperation(**arguments)
    assert "Parameter 'label_attribute' must be informed for task" in str(
        val_err.value)
