import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from juicer.explainable_ai.interpretability import LinearRegressionInterpretation


def test_linear_regression_interpretation(create_iris):
    lreg = LinearRegression()
    lreg.fit(create_iris.data, create_iris.target)
    argument = {'feature_importance': {'n_feature': 1}}
    uai = LinearRegressionInterpretation(argument, lreg, data_source=create_iris.data)
    uai.generate_arguments()
    d_args = uai.generated_args_dict['feature_importance']
    assert len(d_args[0]) == 1
    assert len(d_args[1]) == 1
    assert max(lreg.coef_) == d_args[0]
    assert np.array_equal(uai.feature_importance, lreg.coef_)
    assert np.array_equal(uai.feature_names, create_iris.data.columns)


def test_p_value(create_iris):
    df = pd.concat([create_iris.data, create_iris.target], axis=1, ignore_index=True)
    lreg = LinearRegression()
    lreg.fit(create_iris.data.values, create_iris.target.values)
    info_arg = {'p_value': {'intercept': [1]}}
    uai = LinearRegressionInterpretation(info_arg, lreg, df)
    uai.generate_arguments()

    print(uai.generated_args_dict)




