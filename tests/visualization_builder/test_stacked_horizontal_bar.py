from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from tests.scikit_learn.fixtures import *

import pytest

# Stacked horizontal bar


@pytest.fixture
def get_arguments():
    return {
        "parameters": {
            "type": "stacked-horizontal-bar",
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

    return data, layout


# test 'alignmentgroup' field for each serie
def test_alignmentgroup(generated_chart):
    data, _ = generated_chart
    alignmentgroups = [trace.get("alignmentgroup") for trace in data]
    expected_alignmentgroups = ["True", "True"]
    assert (
        alignmentgroups == expected_alignmentgroups
    ), "incorrect 'alignmentgroup' value for one or multiple series"


# test 'hovertemplate' field for each serie


def test_hovertemplate(generated_chart):
    data, _ = generated_chart
    hovertemplates = [trace.get("hovertemplate") for trace in data]
    expected_hovertemplates = [
        "Série=aggr_0<br>value=%{x}<br>class=%{y}<extra></extra>",
        "Série=aggr_1<br>value=%{x}<br>class=%{y}<extra></extra>",
    ]
    assert (
        hovertemplates == expected_hovertemplates
    ), "incorrect hoverplate for one or multiple series"


# test 'legendgroup' field for each serie


def test_legendgroup(generated_chart):
    data, _ = generated_chart
    legendgroups = [trace.get("legendgroup") for trace in data]
    expected_legendgroups = ["aggr_0", "aggr_1"]
    assert (
        legendgroups == expected_legendgroups
    ), "incorret legendgroup for one or multiple series"


# test 'marker' field for each serie


def test_marker(generated_chart):
    data, _ = generated_chart
    markers = [trace.get("marker") for trace in data]
    for marker in markers:
        assert marker is not None, "'marker' field not found in one or multiple series"
        assert "color" in marker, "'color' field not found in 'marker'"
        assert "pattern" in marker, "'pattern' field not found in 'marker'"


# test 'name' field for each serie


def test_name(generated_chart):
    data, _ = generated_chart
    names = [trace.get("name") for trace in data]
    expected_names = ["aggr_0", "aggr_1"]
    assert names == expected_names, "incorret name for one or multiple series"


# test 'offsetgroup' field for each serie


def test_offsetgroup(generated_chart):
    data, _ = generated_chart
    offsetgroups = [trace.get("offsetgroup") for trace in data]
    expected_offsetgroups = ["aggr_0", "aggr_1"]
    assert (
        offsetgroups == expected_offsetgroups
    ), "incorrect 'offsetgroup' value for one or multiple series"


# test 'orientation' field for each serie


def test_orientation(generated_chart):
    data, _ = generated_chart
    orientations = [trace.get("orientation") for trace in data]
    expected_orientations = ["h", "h"]
    assert (
        orientations == expected_orientations
    ), "incorrect orientation value for one or multiple series"


# test 'showlegend' field for each serie


def test_showlegend(generated_chart):
    data, _ = generated_chart
    showlegends = [trace.get("showlegend") for trace in data]
    expected_showlegends = [True, True]
    assert (
        showlegends == expected_showlegends
    ), "incorrect showlegend value for one or multiple series"


# test 'textposition' field for each serie


def test_textposition(generated_chart):
    data, _ = generated_chart
    textpositions = [trace.get("textposition") for trace in data]
    expected_textpositions = ["auto", "auto"]
    assert (
        textpositions == expected_textpositions
    ), "incorrect 'textposition' value for one or multiple series"


# test 'x' field for each serie


def test_x(generated_chart):
    data, _ = generated_chart
    xs = [trace.get("x") for trace in data]
    expected_xs = [[5.005999999999999, 5.936, 6.587999999999998], [1.464, 4.26, 5.552]]
    assert xs == expected_xs, "incorrect 'x' values for one or multiple series"


# test 'xaxis' field for each serie


def test_xaxis(generated_chart):
    data, _ = generated_chart
    xaxes = [trace.get("xaxis") for trace in data]
    expected_xaxes = ["x", "x"]
    assert (
        xaxes == expected_xaxes
    ), "incorrect 'xaxis' values for one or multiple series"


# test 'y' field for each serie


def test_y(generated_chart):
    data, _ = generated_chart
    ys = [trace.get("y") for trace in data]
    expected_ys = [
        ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
    ]
    assert ys == expected_ys, "incorrect 'y' values for one or multiple series"


# test 'yaxis' field for each serie


def test_yaxis(generated_chart):
    data, _ = generated_chart
    yaxes = [trace.get("yaxis") for trace in data]
    expected_yaxes = ["y", "y"]
    assert (
        yaxes == expected_yaxes
    ), "incorrect 'yaxis' values for one or multiple series"


# test 'type' field for each serie


def test_type(generated_chart):
    data, _ = generated_chart
    types = [trace.get("type") for trace in data]
    expected_types = ["bar", "bar"]
    assert types == expected_types, "incorrect 'type' values for one or multiple series"
