import numpy as np
import pytest

from juicer.explainable_ai.understand_ai import Understanding


def test_feature_names(create_iris):

    argument = {'feature_importance': {'n_feature': 3}}

    uai = Understanding(argument, None, create_iris.data)
    assert np.array_equal(uai.feature_names, create_iris.data.columns)

    uai2 = Understanding(argument, None, create_iris.target)
    assert uai2.feature_names[0] == create_iris.target.name

    uai3 = Understanding(argument, None, create_iris.data, feature_names=['a', 'b', 'c'])
    assert np.array_equal(uai3.feature_names, ['a', 'b', 'c'])


def test_argumets_used(create_iris):
    argument = {'feature_importance': {'n_feature': 3}}
    uai = Understanding(argument, None, create_iris.data)
    assert uai.arguments_used['feature_importance'] == argument['feature_importance']

    with pytest.raises(ValueError) as e_info:
        agt = {'dont_exist': {'n_feature': 3}}
        uai2 = Understanding(agt, None, create_iris.data)
