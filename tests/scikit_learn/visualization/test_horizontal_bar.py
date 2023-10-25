from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import pytest
from tests.scikit_learn.fixtures import *

# Horizontal Bar


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
                },
                {
                    "attribute": "petallength",
                    "aggregation": "AVG",
                    "displayOn": "left",
                    "decimal_places": 2,
                    "strokeSize": 0,
                    "enabled": True,
                },
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
    #
    return data, layout


# test 'hovertemplate' field
def test_data_hovertemplate(generated_chart):
    data, layout = generated_chart
    for trace in data:
        hovertemplate = trace.get("hovertemplate")
        assert (
            hovertemplate is not None
        ), "'hovertemplate' field not found in data object"
        assert "%{x}" in hovertemplate, "'%{x}' field not found in 'hovertemplate'"
        assert "%{y}" in hovertemplate, "'%{y}' field not found in 'hovertemplate'"
        assert (
            "%{marker.size}" in hovertemplate
        ), "'%{marker.size}' field not found in 'hovertemplate'"


# test 'legendgroup' field


def test_data_legendgroup(generated_chart):
    data, layout = generated_chart
    for trace in data:
        legendgroup = trace.get("legendgroup")
        assert legendgroup is not None, "'legendgroup' field not found in data object"


# test 'marker' field


def test_data_marker(generated_chart):
    data, layout = generated_chart
    for trace in data:
        marker = trace.get("marker")
        assert marker is not None, "'marker' field not found in data object"
        assert "color" in marker, "'color' field not found in 'marker'"
        assert "size" in marker, "'size' field not found in 'marker'"
        assert "sizemode" in marker, "'sizemode' field not found in 'marker'"
        assert "sizeref" in marker, "'sizeref' field not found in 'marker'"
        assert "symbol" in marker, "'symbol' field not found in 'marker'"


# test 'mode' field


def test_data_mode(generated_chart):
    data, layout = generated_chart
    for trace in data:
        mode = trace.get("mode")
        assert mode is not None, "'mode' field not found in data object"


# test 'name' field


def test_data_name(generated_chart):
    data, layout = generated_chart
    for trace in data:
        name = trace.get("name")
        assert name is not None, "'name' field not found in data object"


# test 'orientation' field


def test_data_orientation(generated_chart):
    data, layout = generated_chart
    for trace in data:
        orientation = trace.get("orientation")
        assert orientation is not None, "'orientation' field not found in data object"


# test 'showlegend' field


def test_data_showlegend(generated_chart):
    data, layout = generated_chart
    for trace in data:
        showlegend = trace.get("showlegend")
        assert showlegend is not None, "'showlegend' field not found in data object"


# layout

# test 'template' field


def test_layout_template(generated_chart):
    data, layout = generated_chart
    template = layout.get("template")
    assert template is not None, "'template' field not found in layout object"
    assert "data" in template, "'data' field not found in 'template'"
    assert "scatter" in template["data"], "'scatter' field not found in 'template.data'"


# test 'xaxis' field


def test_layout_xaxis(generated_chart):
    data, layout = generated_chart
    xaxis = layout.get("xaxis")
    assert xaxis is not None, "'xaxis' field not found in layout object"
    assert "anchor" in xaxis, "'anchor' field not found in 'xaxis'"
    assert "domain" in xaxis, "'domain' field not found in 'xaxis'"
    assert "title" in xaxis, "'title' field not found in 'xaxis'"
    assert "text" in xaxis["title"], "'text' field not found in 'xaxis.title'"
    assert "categoryorder" in xaxis, "'categoryorder' field not found in 'xaxis'"


# test 'yaxis' field


def test_layout_yaxis(generated_chart):
    data, layout = generated_chart
    yaxis = layout.get("yaxis")
    assert yaxis is not None, "'yaxis' field not found in layout object"
    assert "anchor" in yaxis, "'anchor' field not found in 'yaxis'"
    assert "domain" in yaxis, "'domain' field not found in 'yaxis'"
    assert "title" in yaxis, "'title' field not found in 'yaxis'"
    assert "text" in yaxis["title"], "'text' field not found in 'yaxis.title'"
    assert "showgrid" in yaxis, "'showgrid' field not found in 'yaxis'"
    assert "gridcolor" in yaxis, "'gridcolor' field not found in 'yaxis'"
    assert "visible" in yaxis, "'visible' field not found in 'yaxis'"
    assert "tickformat" in yaxis, "'tickformat' field not found in 'yaxis'"


# test 'legend' field


def test_layout_legend(generated_chart):
    data, layout = generated_chart
    legend = layout.get("legend")
    assert legend is not None, "'legend' field not found in layout object"
    assert "title" in legend, "'title' field not found in 'legend'"
    assert "text" in legend["title"], "'text' field not found in 'legend.title'"
    assert "tracegroupgap" in legend, "'tracegroupgap' field not found in 'legend'"
    assert "itemsizing" in legend, "'itemsizing' field not found in 'legend'"
    assert legend["itemsizing"] == "constant", "'itemsizing' value should be 'constant'"


# test 'margin' field


def test_layout_margin(generated_chart):
    data, layout = generated_chart
    margin = layout.get("margin")
    assert margin is not None, "'margin' field not found in layout object"
    assert "t" in margin, "'t' field not found in 'margin'"
    assert "l" in margin, "'l' field not found in 'margin'"
    assert "r" in margin, "'r' field not found in 'margin'"
    assert "b" in margin, "'b' field not found in 'margin'"
