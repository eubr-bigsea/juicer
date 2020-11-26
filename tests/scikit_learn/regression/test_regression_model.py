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
    'RidgeCV()',
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

        arguments = {
            f'parameters': {'multiplicity': {'train input data': 1},
                            'features': [['sepallength', 'sepalwidth']],
                            'label': [['sepalwidth']]},
            'named_inputs': {
                'algorithm': func,
                'train input data': 'df'
            },
            'named_outputs': {
                'output data': 'out'
            }
        }
        instance = RegressionModelOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'df': df})

        algorithm = eval(func)
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
