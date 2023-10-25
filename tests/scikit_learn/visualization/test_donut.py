from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import pytest

from tests.scikit_learn.fixtures import *

# Donut


@pytest.fixture
def get_arguments():
    return {
        "parameters": {
            "type": "donut",
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
                    "attribute": "class",
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
            "y": [
                {
                    "attribute": "class",
                    "aggregation": "COUNT",
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


def test_layout_domain(generated_chart):
    data, layout = generated_chart
    domain = data[0].get("domain")
    assert domain is not None, "'domain' field not found in layout object"
    assert "x" in domain, "'x' field not found in 'domain'"
    assert "y" in domain, "'y' field not found in 'domain'"
    assert domain["x"] == [0.0, 1.0], "incorrect 'x' field value in 'domain'"
    assert domain["y"] == [0.0, 1.0], "incorrect 'y' field value in 'domain'"


# test 'hole' field


def test_layout_hole(generated_chart):
    data, layout = generated_chart
    hole = data[0].get("hole")
    assert hole is not None, "'hole' field not found in layout object"
    assert hole == 0.3, "incorrect 'hole' field value"


# test 'hovertemplate' field


def test_layout_hovertemplate(generated_chart):
    data, layout = generated_chart
    hovertemplate = data[0].get("hovertemplate")
    assert hovertemplate is not None, "'hovertemplate' field not found in data object"
    assert (
        hovertemplate == "class=%{label}<br>count(class)=%{value}<extra></extra>"
    ), "incorrect 'hovertemplate' field value"


# test 'labels' field


def test_layout_labels(generated_chart):
    data, layout = generated_chart
    labels = data[0].get("labels")
    assert labels is not None, "'labels' field not found in data] object"
    assert labels == [
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica",
    ], "incorrect 'labels' field value"


# test 'legendgroup' field


def test_layout_legendgroup(generated_chart):
    data, layout = generated_chart
    legendgroup = data[0].get("legendgroup")
    assert legendgroup is not None, "'legendgroup' field not found in data] object"
    assert legendgroup == "", "incorrect 'legendgroup' field value"


# test 'name' field


def test_layout_name(generated_chart):
    data, layout = generated_chart
    name = data[0].get("name")
    assert name is not None, "'name' field not found in data object"
    assert name == "", "incorrect 'name' field value"


# test 'showlegend' field


def test_layout_showlegend(generated_chart):
    data, layout = generated_chart
    showlegend = data[0].get("showlegend")
    assert showlegend is not None, "'showlegend' field not found in data object"
    assert showlegend == True, "incorrect 'showlegend' field value"


# test 'values' field


def test_layout_values(generated_chart):
    data, layout = generated_chart
    values = data[0].get("values")
    assert values is not None, "'values' field not found in data object"
    assert values == [50.0, 50.0, 50.0], "incorrect 'values' field value"


# test 'type' field


def test_layout_type(generated_chart):
    data, layout = generated_chart
    chart_type = data[0].get("type")
    assert chart_type is not None, "'type' field not found in data object"
    assert chart_type == "pie", "incorrect 'type' field value"


# layout


# test 'template' field
def test_layout_template(generated_chart):
    data, layout = generated_chart
    template = layout.get("template")
    assert template is not None, "'template' field not found in layout object"
    assert "data" in template, "'data' field not found in 'template'"
    assert (
        "scatter" in template["data"]
    ), "'scatter' field not found in 'template['data']"
    assert (
        template["data"]["scatter"][0]["type"] == "scatter"
    ), "incorrect 'type' field value in 'scatter'"


# test 'legend' field


def test_layout_legend(generated_chart):
    data, layout = generated_chart
    legend = layout.get("legend")
    assert legend is not None, "'legend' field not found in layout object"
    assert "tracegroupgap" in legend, "'tracegroupgap' field not found in 'legend'"


# test 'margin' field


def test_layout_margin(generated_chart):
    data, layout = generated_chart
    margin = layout.get("margin")
    assert margin is not None, "'margin' field not found in layout object"
    assert "t" in margin, "'t' field not found in 'margin'"
    assert "l" in margin, "'l' field not found in 'margin'"
    assert "r" in margin, "'r' field not found in 'margin'"
    assert "b" in margin, "'b' field not found in 'margin'"


# test 'piecolorway' field


def test_layout_piecolorway(generated_chart):
    data, layout = generated_chart
    piecolorway = layout.get("piecolorway")
    assert piecolorway is not None, "'piecolorway' field not found in layout object"
    expected_colors = [
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
    ]
    assert piecolorway == expected_colors, "incorrect 'piecolorway' field value"


# test 'extendpiecolors' field


def test_layout_extendpiecolors(generated_chart):
    data, layout = generated_chart
    extendpiecolors = layout.get("extendpiecolors")
    assert (
        extendpiecolors is not None
    ), "'extendpiecolors' field not found in layout object"
    assert extendpiecolors == True, "incorrect 'extendpiecolors' field value"


# test 'xaxis' field


def test_layout_xaxis(generated_chart):
    data, layout = generated_chart
    xaxis = layout.get("xaxis")
    assert xaxis is not None, "'xaxis' field not found in layout object"
    assert "categoryorder" in xaxis, "'categoryorder' field not found in 'xaxis'"
    assert (
        xaxis["categoryorder"] == "trace"
    ), "incorrect 'categoryorder' field value in 'xaxis'"
