from sklearn.tree import DecisionTreeClassifier

from juicer.explainable_ai.plot_generate_xai import PltGenerate


def test_plot_feature_importance():
    argument = {'feature_importance': ([1, 2, 3], ['a', 'b', 'c'])}
    plt_xai = PltGenerate(argument)
    plt_xai.create_plots()


def test_plot_forest_importance():
    argument = {
        'forest_importance': ([1, 2, 3], [0.25, 0.5, 0.75], ['A', 'B', 'C']),
        'feature_importance': ([1, 2, 3], ['a', 'b', 'c'])
    }
    plt_xai = PltGenerate(argument)
    plt_xai.create_plots()


def test_plot_dt_surface(create_iris):
    dtcls = DecisionTreeClassifier()
    dtcls.fit(create_iris.data, create_iris.target)
    arg = {'dt_surface': {'max_deep': 4, 'model': dtcls, 'feature_names': create_iris.data.columns}}
    plt_xai = PltGenerate(arg)
    plt_xai.create_plots()
