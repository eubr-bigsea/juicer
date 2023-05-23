import base64
import io

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from juicer.explainable_ai.explainability import GeneticProgrammingExplainer
from juicer.explainable_ai.plot_generate_xai import PltGenerate


def test_gp_solver(create_diabetes):
    lreg = LinearRegression()
    lreg.fit(create_diabetes.data, create_diabetes.target)
    data = pd.concat([create_diabetes.data, create_diabetes.target], axis=1)
    gpx = GeneticProgrammingExplainer({}, lreg, data)

    assert gpx.gp_solver.__class__.__name__ == "SymbolicRegressor"


def test_feature_importance(create_diabetes):
    lreg = LinearRegression()
    lreg.fit(create_diabetes.data.values, create_diabetes.target.values)
    data = pd.concat([create_diabetes.data, create_diabetes.target], axis=1)
    instance = [0.005383, -0.044642, -0.036385,  0.021872,  0.003935,  0.015596,  0.008142, -0.002592, -0.031991, -0.046641]
    gpx = GeneticProgrammingExplainer({'feature_importance': {'instance': instance}}, lreg, data)
    gpx.generate_arguments()

    assert np.array_equal(gpx.feature_names, data.columns)
    assert gpx.gp_solver.__class__.__name__ == "SymbolicRegressor"


def test_feature_importance(create_diabetes):
    lreg = LinearRegression()
    lreg.fit(create_diabetes.data.values, create_diabetes.target.values)
    data = pd.concat([create_diabetes.data, create_diabetes.target], axis=1)
    instance = [0.005383, -0.044642, -0.036385,  0.021872,  0.003935,  0.015596,  0.008142, -0.002592, -0.031991, -0.046641]
    gpx = GeneticProgrammingExplainer({'feature_importance': {'instance': instance}}, lreg, data)
    gpx.generate_arguments()
    print(gpx.generated_args_dict)
    t = gpx.explainer.show_tree(is_base64=True)

    decoded_data = base64.b64decode(t)
    image_stream = io.BytesIO(decoded_data)
    image = plt.imread(image_stream)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig('tree.png')

    plt_xai = PltGenerate(gpx.generated_args_dict)
    p = plt_xai.create_plots()

    decoded_data = base64.b64decode(p)
    image_stream = io.BytesIO(decoded_data)
    image = plt.imread(image_stream)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig('partial_derivatives.png')




