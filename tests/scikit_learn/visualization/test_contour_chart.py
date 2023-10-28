from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from plotly import graph_objects as go

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors


# histogram2dcontour


df = util.iris2_polars()


@pytest.fixture
def get_df():
    return util.iris2_polars()


@pytest.fixture
def get_arguments():
    return {
        "parameters": {
            "type": "histogram2dcontour",
            "display_legend": "HIDE",
            "x": [
                {
                    "binning": "EQUAL_INTERVAL",
                    "bins": 20,
                    "binSize": 10,
                    "emptyBins": "ZEROS",
                    "multiplier": None,
                    "decimal_places": 2,
                    "prefix": None,
                    "suffix": None,
                    "label": None,
                    "max_displayed": None,
                    "group_others": True,
                    "sorting": "NATURAL",
                    "attribute": "PetalLengthCm",
                }
            ],
            "color_scale": ["#0000ff", "#ff0000"],
            "y": [
                {
                    "attribute": "PetalWidthCm",
                    "aggregation": "MIN",
                    "compute": None,
                    "displayOn": "left",
                    "multiplier": None,
                    "decimal_places": 2,
                    "prefix": None,
                    "suffix": None,
                    "label": None,
                    "strokeSize": 0,
                    "stroke": None,
                    "color": None,
                    "marker": None,
                    "enabled": True,
                }
            ],
            "x_axis": {
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
            "import json",
            instance.generate_code(),
        ]
    )
    result = util.execute(code, vis_globals)
    generated_chart = result.get("d")
    data = generated_chart["data"]
    layout = generated_chart["layout"]

    return data, layout


# Test to verify the 'contours' field
def test_data_contours(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]
    contours = histogram2dcontour_data.get("contours")
    assert contours is not None, "Field 'contours' not found in the data object"
    expected_contours = {"coloring": "fill", "showlabels": True}
    assert contours == expected_contours, "Incorrect values for 'contours' field"


# Test to verify the 'line' field
def test_data_line(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]
    line = histogram2dcontour_data.get("line")
    assert line is not None, "Field 'line' not found in the data object"


# Test to verify the 'name' field
def test_data_name(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]
    name = histogram2dcontour_data.get("name")
    assert name is not None, "Field 'name' not found in the data object"
    assert name == "", "Incorrect value for 'name' field"


# Test to verify the 'showlegend' field
def test_data_showlegend(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]
    showlegend = histogram2dcontour_data.get("showlegend")
    assert showlegend is not None, "Field 'showlegend' not found in the data object"
    assert showlegend == False, "Incorrect value for 'showlegend' field"


# Test to verify the 'x' field
def test_data_x(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]
    x = histogram2dcontour_data.get("x")
    assert x is not None, "Field 'x' not found in the data object"
    expected_x = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    assert x == expected_x, "Incorrect values for 'x' field"


# Test to verify the 'y' field
def test_data_y(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]
    y = histogram2dcontour_data.get("y")
    assert y is not None, "Field 'y' not found in the data object"
    expected_y = [
        0.1,
        0.1,
        0.2,
        0.2,
        1.1,
        1.0,
        1.0,
        1.0,
        1.0,
        1.2,
        1.2,
        1.5,
        1.9,
        1.4,
        1.6,
        1.8,
        2.0,
        2.0,
    ]
    assert y == expected_y, "Incorrect values for 'y' field"


# Test to verify the 'colorscale' field
def test_data_colorscale(generated_chart):
    data, layout = generated_chart
    histogram2dcontour_data = data[0]
    colorscale = histogram2dcontour_data.get("colorscale")
    assert colorscale is not None, "Field 'colorscale' not found in the data object"


# Test to verify the 'template' field
def test_layout_template(generated_chart):
    data, layout = generated_chart
    layout_template = layout.get("template")
    assert (
        layout_template is not None
    ), "Field 'template' not found in the layout object"
    expected_template = {"data": {"scatter": [{"type": "scatter"}]}}
    assert layout_template == expected_template, "Incorrect values for 'template' field"


# Test to verify the 'xaxis' field
def test_layout_xaxis(generated_chart):
    data, layout = generated_chart
    layout_xaxis = layout.get("xaxis")
    assert layout_xaxis is not None, "Field 'xaxis' not found in the layout object"
    expected_xaxis = {
        "anchor": "y",
        "domain": [0.0, 1.0],
        "title": {"text": "PetalLengthCm"},
        "categoryorder": "trace",
    }
    assert layout_xaxis == expected_xaxis, "Incorrect values for 'xaxis' field"


# Test to verify the 'yaxis' field
def test_layout_yaxis(generated_chart):
    data, layout = generated_chart
    layout_yaxis = layout.get("yaxis")
    assert layout_yaxis is not None, "Field 'yaxis' not found in the layout object"
    expected_yaxis = {
        "anchor": "x",
        "domain": [0.0, 1.0],
        "title": {"text": "min(PetalWidthCm)"},
    }
    assert layout_yaxis == expected_yaxis, "Incorrect values for 'yaxis' field"


# Test to verify the 'legend' field
def test_layout_legend(generated_chart):
    data, layout = generated_chart
    layout_legend = layout.get("legend")
    assert layout_legend is not None, "Field 'legend' not found in the layout object"
    expected_legend = {"tracegroupgap": 0}
    assert layout_legend == expected_legend, "Incorrect values for 'legend' field"
