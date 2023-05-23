import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from juicer.explainable_ai.interpretability import Interpretation


def test_feature_importance(create_diabetes, create_iris):
    lreg = LinearRegression()
    lreg.fit(create_diabetes.data, create_diabetes.target)

    argument = {'feature_importance': {'n_feature': 3}}
    uai = Interpretation(argument, lreg, feature_names=['a', 'b', 'c'])

    assert np.array_equal(uai.feature_importance, lreg.coef_)

    dtcls = DecisionTreeClassifier()
    dtcls.fit(create_iris.data, create_iris.target)

    uai2 = Interpretation(argument, dtcls, feature_names=['a', 'b', 'c'])

    assert np.array_equal(uai2.feature_importance, dtcls.feature_importances_)
    uai3 = Interpretation(argument, feature_importance=[1, 2, 3], feature_names=['a', 'b', 'c'])

    assert np.array_equal(uai3.feature_importance, [1, 2, 3])


def test_n_feature(create_iris):
    dtcls = DecisionTreeClassifier()
    dtcls.fit(create_iris.data, create_iris.target)
    argument = {'feature_importance': {'n_feature': 1}}
    uai = Interpretation(argument, dtcls, data_source=create_iris.data)
    uai.generate_arguments()
    d_args = uai.generated_args_dict['feature_importance']
    assert len(d_args[0]) == 1
    assert len(d_args[1]) == 1
    assert max(dtcls.feature_importances_) == d_args[0]

