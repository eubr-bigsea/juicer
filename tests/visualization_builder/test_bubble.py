from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import pytest
from tests.scikit_learn.fixtures import *

# Bubble


@pytest.fixture
def get_arguments():
    return {
        "parameters": {
            "type": "bubble",
            "display_legend": "AUTO",
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
                }
            ],
            "palette": [
                "#1F77B4",
                "#FF7F0E",
                "#2CA02C",
                "#D62728",
                "#9467BD",
                "#8C564B",
                "#E377C2",
                "#7F7F7F",
                "#BCBD22",
                "#17BECF",
            ],
            "color_attribute": {
                "id": 5261,
                "name": "class",
                "type": "CHARACTER",
                "size": 15,
                "nullable": False,
                "enumeration": False,
                "feature": False,
                "label": False,
                "key": False,
                "attribute": "class",
                "numeric": False,
                "integerType": False,
            },
            "size_attribute": {
                "id": 5259,
                "name": "petallength",
                "type": "DECIMAL",
                "precision": 2,
                "scale": 1,
                "nullable": False,
                "enumeration": False,
                "feature": False,
                "label": False,
                "key": False,
                "attribute": "petallength",
                "numeric": True,
                "integerType": False,
            },
            "y": [
                {
                    "attribute": "sepallength",
                    "aggregation": "AVG",
                    "displayOn": "left",
                    "decimal_places": 2,
                    "strokeSize": 0,
                    "enabled": True,
                }
            ],
            "x_axis": {
                "logScale": False,
                "display": True,
                "displayLabel": True,
                "decimal_places": 2,
            },
            "y_axis": {
                "logScale": False,
                "display": True,
                "displayLabel": True,
                "decimal_places": 2,
            },
            "subgraph_orientation": "v",
            "task_id": "1",
        },
        "named_inputs": {
            "input data": "iris",
        },
        "named_outputs": {"output data": "out"},
    }


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


# test 'hovertemplate' field


def test_data_hovertemplate(generated_chart):
    data, layout = generated_chart
    for scatter_data in data:
        hovertemplate = scatter_data.get("hovertemplate")
        assert hovertemplate is not None, "'hovertemplate' not found in data object"
        assert "class=" in hovertemplate, "'class=' text not found in 'hovertemplate'"
        assert "%{x}" in hovertemplate, "'%{x}' marker not found in 'hovertemplate'"
        assert "%{y}" in hovertemplate, "'%{y}' marker not found in hovertemplate'"
        assert (
            "petallength=%{marker.size}" in hovertemplate
        ), "'petallength=%{marker.size}' text not found in 'hovertemplate'"
        assert (
            "<extra></extra>" in hovertemplate
        ), "'<extra></extra>' text not found in 'hovertemplate'"


# test 'legendgroup' field


def test_data_legendgroup(generated_chart):
    data, layout = generated_chart
    for scatter_data in data:
        legendgroup = scatter_data.get("legendgroup")
        assert legendgroup is not None, "'legendgroup' not found in data object"


# test 'marker' field


def test_data_marker(generated_chart):
    data, layout = generated_chart
    for scatter_data in data:
        marker = scatter_data.get("marker")
        assert marker is not None, "'marker' not found in data object"
        assert "color" in marker, "'color' field not found in 'marker''"
        assert "size" in marker, "'size' field not found in 'marker''"
        assert "sizemode" in marker, "'sizemode' field not found in 'marker''"
        assert "sizeref" in marker, "'sizeref' field not found in 'marker''"
        assert "symbol" in marker, "'symbol' field not found in 'marker''"


# test 'mode' field


def test_data_mode(generated_chart):
    data, layout = generated_chart
    for scatter_data in data:
        mode = scatter_data.get("mode")
        assert mode is not None, "'mode' not found in data object"
        assert mode == "markers", "incorrect 'mode'field value"


# test 'name' field


def test_data_name(generated_chart):
    data, layout = generated_chart
    for scatter_data in data:
        name = scatter_data.get("name")
        assert name is not None, "'name' not found in data object"


# test 'orientation' field


def test_data_orientation(generated_chart):
    data, layout = generated_chart
    for scatter_data in data:
        orientation = scatter_data.get("orientation")
        assert orientation is not None, "'orientation' not found in data object"
        assert orientation == "v", "incorrect 'orientation' field value"


# test 'showlegend' field


def test_data_showlegend(generated_chart):
    data, layout = generated_chart
    for scatter_data in data:
        showlegend = scatter_data.get("showlegend")
        assert showlegend is not None, "'showlegend' not found in data object"
        assert showlegend == True, "incorrect 'showlegend' field value"


# test 'x' field


def test_data_x(generated_chart):
    data, layout = generated_chart
    for scatter_data in data:
        x = scatter_data.get("x")
        assert x is not None, "'x' not found in data object"


# test 'y' field


def test_data_y(generated_chart):
    data, layout = generated_chart
    for scatter_data in data:
        y = scatter_data.get("y")
        assert y is not None, "'y' not found in data object"


# test 'xaxis' field


def test_data_xaxis(generated_chart):
    data, layout = generated_chart
    for scatter_data in data:
        xaxis = scatter_data.get("xaxis")
        assert xaxis is not None, "'xaxis' not found in data object"


# test 'yaxis' field


def test_data_yaxis(generated_chart):
    data, layout = generated_chart
    for scatter_data in data:
        yaxis = scatter_data.get("yaxis")
        assert yaxis is not None, "'yaxis' not found in data object"


# test 'type' field


def test_data_type(generated_chart):
    data, layout = generated_chart
    for scatter_data in data:
        chart_type = scatter_data.get("type")
        assert chart_type is not None, "'type' not found in data object"
        assert chart_type == "scatter", "incorrect 'type' field value"


# layout

# test 'template' field


def test_layout_template(generated_chart):
    data, layout = generated_chart
    template = layout.get("template")
    assert template is not None, "'template' not found in layout object"


# test 'xaxis' field


def test_layout_xaxis(generated_chart):
    data, layout = generated_chart
    xaxis = layout.get("xaxis")
    assert xaxis is not None, "'xaxis' not found in layout object"
    assert "anchor" in xaxis, "'anchor' field not found in 'xaxis''"
    assert "domain" in xaxis, "'domain' field not found in 'xaxis''"
    assert "title" in xaxis, "'title' field not found in 'xaxis''"
    assert (
        xaxis["title"]["text"] == "petalwidth"
    ), "incorrect 'text' value field in  'title' of 'xaxis'"
    assert "categoryorder" in xaxis, "'categoryorder' field not found in 'xaxis''"


# test 'yaxis'field


def test_layout_yaxis(generated_chart):
    data, layout = generated_chart
    yaxis = layout.get("yaxis")
    assert yaxis is not None, "'yaxis' field not found in layout object"
    assert "anchor" in yaxis, "'anchor' field not found in 'yaxis''"
    assert "domain" in yaxis, "'domain' field not found in 'yaxis''"
    assert "title" in yaxis, "'title' field not found in 'yaxis''"
    assert (
        yaxis["title"]["text"] == "avg(sepallength)"
    ), "incorrect 'text' value field in  'title' of 'yaxis'"
    assert "showgrid" in yaxis, "'showgrid' field not found in 'yaxis''"
    assert "gridcolor" in yaxis, "'gridcolor' field not found in 'yaxis''"
    assert "visible" in yaxis, "'visible' field not found in 'yaxis''"
    assert "tickformat" in yaxis, "'tickformat' field not found in 'yaxis''"


# test 'legend' field


def test_layout_legend(generated_chart):
    data, layout = generated_chart
    legend = layout.get("legend")
    assert legend is not None, "'legend' field not found in layout object"
    assert "title" in legend, "'title' field not found in 'legend''"
    assert (
        legend["title"]["text"] == "class"
    ), "incorrect 'text' value field in  'title' of 'legend'"
    assert "tracegroupgap" in legend, "'tracegroupgap' field not found in 'legend''"
    assert "itemsizing" in legend, "'itemsizing' field not found in 'legend''"


# test 'margin field'


def test_layout_margin(generated_chart):
    data, layout = generated_chart
    margin = layout.get("margin")
    assert margin is not None, "'margin' field not found in layout object"
    assert "t" in margin, "'t' field not found in 'margin''"
    assert "l" in margin, "'l' field not found in 'margin''"
    assert "r" in margin, "'r' field not found in 'margin''"
    assert "b" in margin, "'b' field not found in 'margin''"
