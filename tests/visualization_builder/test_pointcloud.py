from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from plotly import graph_objects as go


import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors


# pointcloud

def to_numeric(s: pl.Series) -> pl.Series:
    try:
        result = s.cast(pl.Int64)
    except pl.exceptions.ComputeError:
        result = s.cast(pl.Float64)
    return result


@pytest.fixture
def get_df():
    df = util.iris_polars()
    '''
    mapa_classes = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
    df = df.select(
        pl.col("sepallength"),
        pl.col("sepalwidth"),
        pl.col("petallength"),
        pl.col("petalwidth"),
        pl.col("class").apply(lambda x: mapa_classes.get(x, 0), pl.Int64()))
    '''
    return df

@pytest.fixture
def get_arguments():
    return {
        "parameters": {
            "type": "pointcloud",
            "display_legend": "LEFT",
            "x": [
                {
                    "binning": None,
                    "bins": 20,
                    "binSize": 10,
                    "emptyBins": "ZEROS",
                    "decimal_places": 2,
                    "group_others": True,
                    "sorting": "NATURAL",
                    "attribute": "petalwidth",
                },
                {
                    "binning": None,
                    "bins": 20,
                    "binSize": 10,
                    "emptyBins": "ZEROS",
                    "multiplier": None,
                    "decimal_places": None,
                    "prefix": None,
                    "suffix": None,
                    "label": None,
                    "max_displayed": None,
                    "group_others": True,
                    "sorting": None,
                    "attribute": "sepalwidth",
                },
            ],
            "palette": [
                "#1b9e77",
                "#d95f02",
                "#7570b3",
                "#e7298a",
                "#66a61e",
                "#e6ab02",
                "#a6761d",
                "#666666",
            ],
            "y": [
                {
                    "attribute": "petallength",
                    "aggregation": "COUNT",
                    "compute": None,
                    "displayOn": "left",
                    "multiplier": 10,
                    "decimal_places": 3,
                    "prefix": "",
                    "suffix": None,
                    "label": "TESTE",
                    "strokeSize": 0,
                    "stroke": None,
                    "color": "#da1616",
                    "marker": None,
                    "custom_color": False,
                    "line_color": "#141414",
                    "enabled": True,
                }
            ],
            "x_axis": {
                "lowerBound": "1",
                "upperBound": "10",
                "logScale": True,
                "display": False,
                "displayLabel": False,
                "label": "TESTE",
                "multiplier": None,
                "decimal_places": 2,
                "prefix": None,
                "suffix": None,
            },
            "y_axis": {
                "lowerBound": None,
                "upperBound": None,
                "logScale": False,
                "display": True,
                "displayLabel": True,
                "label": None,
                "multiplier": None,
                "decimal_places": 2,
                "prefix": None,
                "suffix": None,
            },
            "task_id": "0",
        },
        "named_inputs": {
            "input data": "iris",
        },
        "named_outputs": {"output data": "out"},
    }

def emit_event(*args, **kwargs):
    print(args, kwargs)


@pytest.fixture
def generated_chart(get_arguments, get_df):
    instance = VisualizationOperation(**get_arguments)
    vis_globals = dict(iris=get_df, emit_event=util.emit_event)
    code = "\n".join(
        [
            "import plotly.graph_objects as go",
            "import plotly.express as px",
            "import polars as pl",
            "import json",
            instance.generate_code(),
        ]
    )

    result = util.execute(code, vis_globals)
    generated_chart = result.get("d")
    data = generated_chart["data"]
    layout = generated_chart["layout"]
    return data, layout


# Data tests
# Test to verify the 'mode' field
def test_data_mode(generated_chart):
    data, layout = generated_chart
    scatter_data = data[0]
    mode = scatter_data.get("mode")
    assert mode is not None, "Field 'mode' not found in data"
    assert mode == "markers", "Incorrect value for 'mode' field"


# Test to verify the 'x' field
def test_data_x(generated_chart):
    data, layout = generated_chart
    scatter_data = data[0]
    x = scatter_data.get("x")
    assert x is not None, "Field 'x' not found in data"


# Test to verify the 'y' field
def test_data_y(generated_chart):
    data, layout = generated_chart
    scatter_data = data[0]
    y = scatter_data.get("y")
    assert y is not None, "Field 'y' not found in data"


# Test to verify the 'z' field
def test_data_z(generated_chart):
    data, layout = generated_chart
    scatter_data = data[0]
    z = scatter_data.get("z")
    assert z is not None, "Field 'z' not found in data"


# Layout tests
# Test to verify the 'template' field
def test_layout_template(generated_chart):
    data, layout = generated_chart
    template = layout.get("template")
    assert template is not None, "Field 'template' not found in layout"


# Test to verify the 'scene' field in the layout object
def test_layout_scene(generated_chart):
    data, layout = generated_chart
    scene = layout.get("scene")
    assert scene is not None, "Field 'scene' not found in layout"


# Test to verify the 'legend' field
def test_layout_legend(generated_chart):
    data, layout = generated_chart
    legend = layout.get("legend")
    assert legend is not None, "Field 'legend' not found in layout"


# Test to verify the 'margin' field
def test_layout_margin(generated_chart):
    data, layout = generated_chart
    margin = layout.get("margin")
    assert margin is not None, "Field 'margin' not found in layout"


# Test to verify the 'xaxis' field
def test_layout_xaxis(generated_chart):
    data, layout = generated_chart
    xaxis = layout.get("xaxis")
    assert xaxis is not None, "Field 'xaxis' not found in layout"
