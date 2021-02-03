from tests.scikit_learn import util
from juicer.scikit_learn.regression_operation import RegressionModelOperation
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import *

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

functions = [
    # 'RidgeCV()', # TODO: check it again
    'LinearRegression()',
    'Ridge()',
    'Lasso()',
    'ElasticNet()',
    'ElasticNetCV()',
    'LarsCV()',
    'Lars()',
    'LassoLars()',
    'OrthogonalMatchingPursuit()',
    'BayesianRidge()',
    'ARDRegression()',
    'RANSACRegressor()',
    'TheilSenRegressor()',
    'HuberRegressor()',
    'SGDRegressor()',
    'PassiveAggressiveRegressor()',
    'RANSACRegressor()',
    'TheilSenRegressor()'
]

approx_check = [
    'SGDRegressor()',
    'PassiveAggressiveRegressor()',
    'DecisionTreeRegressor()'
]


# RegressionModel
#
#
# # # # # # # # # # Success # # # # # # # # # #
def test_regression_model_various_models_success():
    for func in functions:
        df = util.iris(['sepallength', 'sepalwidth'], size=10)
        test_df = df.copy()
        algorithm = eval(func)

        arguments = {
            f'parameters': {'multiplicity': {'train input data': 1},
                            'features': [['sepallength', 'sepalwidth']],
                            'label': [['sepalwidth']]},
            'named_inputs': {
                'algorithm': algorithm,
                'train input data': 'df'
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        instance = RegressionModelOperation(**arguments)
        instance.transpiler_utils.add_import(
                "from sklearn.linear_model import *")
        result = util.execute(util.get_complete_code(instance),
                              {'df': df})

        test_out = test_df
        X_train = util.get_X_train_data(df, ['sepallength', 'sepalwidth'])
        y = util.get_label_data(df, ['sepalwidth'])
        if 'IsotonicRegression' in str(algorithm):
            X_train = np.ravel(X_train)
        algorithm.fit(X_train, y)
        test_out['prediction'] = algorithm.predict(X_train).tolist()

        if func in approx_check:
            for col in result['out'].columns:
                for idx in result['out'].index:
                    assert result['out'].loc[idx, col] == pytest.approx(
                        test_out.loc[idx, col], 0.1)
        else:
            assert result['out'].equals(test_out)


def test_regression_model_no_output_implies_no_code_success():
    arguments = {
        f'parameters': {'multiplicity': {'train input data': 1},
                        'features': [['sepallength', 'sepalwidth']],
                        'label': [['sepalwidth']]},
        'named_inputs': {
            'algorithm': 'LinearRegression()',
            'train input data': 'df'
        },
        'named_outputs': {
        }
    }
    instance = RegressionModelOperation(**arguments)
    assert instance.generate_code() is None


def test_regression_model_missing_input_implies_no_code_success():
    arguments = {
        f'parameters': {'multiplicity': {'train input data': 1},
                        'features': [['sepallength', 'sepalwidth']],
                        'label': [['sepalwidth']]},
        'named_inputs': {
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RegressionModelOperation(**arguments)
    assert instance.generate_code() is None


# # # # # # # # # # Fail # # # # # # # # # #
def test_regression_model_missing_features_label_params_fail():
    arguments = {
        f'parameters': {'multiplicity': {'train input data': 1}},
        'named_inputs': {
            'algorithm': 'LinearRegression()',
            'train input data': 'df'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    with pytest.raises(ValueError) as val_err:
        RegressionModelOperation(**arguments)
    assert "Parameters 'features' and 'label' must be informed for task" in \
           str(val_err.value)


def test_regression_model_missing_multiplicity_param_fail():
    df = util.iris(['sepallength', 'sepalwidth'], size=10)
    arguments = {
        f'parameters': {'features': [['sepallength', 'sepalwidth']],
                        'label': [['sepalwidth']]},
        'named_inputs': {
            'algorithm': 'LinearRegression()',
            'train input data': 'df'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    instance = RegressionModelOperation(**arguments)
    with pytest.raises(KeyError) as key_err:
        util.execute(util.get_complete_code(instance),
                     {'df': df})
    assert 'multiplicity' in str(key_err.value)
