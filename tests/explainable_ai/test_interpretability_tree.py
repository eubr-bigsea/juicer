import numpy as np
from sklearn.tree import DecisionTreeClassifier

from juicer.explainable_ai.interpretability import TreeInterpretation


def test_tree_interpretation(create_iris):
    dtcls = DecisionTreeClassifier()
    dtcls.fit(create_iris.data, create_iris.target)
    argument = {'feature_importance': {'n_feature': 1}}
    uai = TreeInterpretation(argument, dtcls, data_source=create_iris.data)
    uai.generate_arguments()
    d_args = uai.generated_args_dict['feature_importance']
    assert len(d_args[0]) == 1
    assert len(d_args[1]) == 1
    assert max(dtcls.feature_importances_) == d_args[0]
    assert np.array_equal(uai.feature_importance, dtcls.feature_importances_)
    assert np.array_equal(uai.feature_names, create_iris.data.columns)


def test_tree_surface(create_iris):
    dtcls = DecisionTreeClassifier()
    dtcls.fit(create_iris.data, create_iris.target)
    argument = {'dt_surface': {'max_deep': 4}}
    uai = TreeInterpretation(argument, dtcls, data_source=create_iris.data)
    uai.generate_arguments()
    d_args = uai.generated_args_dict['dt_surface']
    assert np.array_equal(d_args['feature_names'], create_iris.data.columns)
