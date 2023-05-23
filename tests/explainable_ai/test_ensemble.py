from sklearn.ensemble import RandomForestClassifier

from juicer.explainable_ai.interpretability import EnsembleInterpretation
import numpy as np


def test_ensemble_interpretation(create_iris):
    rf_cls = RandomForestClassifier()
    rf_cls.fit(create_iris.data, create_iris.target)
    argument = {'feature_importance': {'n_feature': 1}, 'forest_importance': {'n_feature': 2}}
    uai = EnsembleInterpretation(argument, rf_cls, data_source=create_iris.data)
    uai.generate_arguments()
    d_args = uai.generated_args_dict['feature_importance']
    assert len(d_args[0]) == 1
    assert len(d_args[1]) == 1
    assert max(rf_cls.feature_importances_) == d_args[0]

    assert np.array_equal(uai.feature_importance, rf_cls.feature_importances_)

    assert np.array_equal(uai.feature_names, create_iris.data.columns)

    gump = uai.generated_args_dict['forest_importance']
    assert len(gump) == 3
    assert max(gump[0]) == d_args[0]