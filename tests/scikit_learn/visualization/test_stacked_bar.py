import numpy as np
import pytest

from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from tests.scikit_learn import util
from tests.scikit_learn.fixtures import *

# Stacked bar


@pytest.fixture
def get_arguments():
    return {
        'parameters': {
            'type': 'stacked-bar',
            'display_legend': "AUTO",
            "x": [{
                "binning": None,
                "bins": 20,
                "binSize": 10,
                "emptyBins": "ZEROS",
                "decimal_places": 2,
                "group_others": True,
                "sorting": "NATURAL",
                "attribute": "class",

            }],
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
                "#17BECF"
            ],
            "y": [{
                "attribute": "sepallength",
                "aggregation": "AVG",
                "displayOn": "left",
                "decimal_places": 2,
                "strokeSize": 0,
                "enabled": True
            }, {
                "attribute": "petallength",
                "aggregation": "AVG",
                "displayOn": "left",
                "decimal_places": 2,
                "strokeSize": 0,
                "enabled": True

            }],
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
            'subgraph_orientation': "v",

            "task_id": "1"
        },
        'named_inputs': {
            'input data': "iris",
        },
        'named_outputs': {
            'output data': 'out'
        }
    }


@pytest.fixture
def generated_chart(get_arguments, get_df):
    instance = VisualizationOperation(**get_arguments)
    vis_globals = dict(iris=get_df, emit_event=util.emit_event)
    code = '\n'.join(["import plotly.graph_objects as go",
                      "import plotly.express as px", "import json", instance.generate_code(), ])
    result = util.execute(code, vis_globals)
    generated_chart = result.get('d')
    data = generated_chart['data']
    layout = generated_chart['layout']
    #
    return data, layout


# test 'hovertemplate' field for each class
def test_hovertemplate(generated_chart):
    data, layout = generated_chart
    hovertemplates = [trace.get('hovertemplate') for trace in data]
    expected_hovertemplates = [
        'Série=aggr_0<br>class=%{x}<br>value=%{y}<extra></extra>',
        'Série=aggr_1<br>class=%{x}<br>value=%{y}<extra></extra>'
    ]
    assert hovertemplates == expected_hovertemplates, "incorrect hovertemplate for one or multiple classes"

# test 'legendgroup' field for each class


def test_legendgroup(generated_chart):
    data, layout = generated_chart
    legendgroups = [trace.get('legendgroup') for trace in data]
    expected_legendgroups = ['aggr_0', 'aggr_1']
    assert legendgroups == expected_legendgroups, "incorrect legendgroup for one or multiple classes"

# test 'marker' field for each class


def test_marker(generated_chart):
    data, layout = generated_chart
    markers = [trace.get('marker') for trace in data]
    for marker in markers:
        assert marker is not None, "'marker' field not found in one or multiple classes"
        assert 'color' in marker, "'color' field not found in 'marker'"
        assert 'pattern' in marker, "'pattern' field not found in 'marker'"

# test 'name' field for each class


def test_name(generated_chart):
    data, layout = generated_chart
    names = [trace.get('name') for trace in data]
    expected_names = ['aggr_0', 'aggr_1']
    assert names == expected_names, "incorrect name for one or multiple classes"

# test 'orientation' field for each class


def test_orientation(generated_chart):
    data, layout = generated_chart
    orientations = [trace.get('orientation') for trace in data]
    expected_orientations = ['v', 'v']
    assert orientations == expected_orientations, "incorrect orientation for one or multiple classes"

# test 'showlegend' field for each class


def test_showlegend(generated_chart):
    data, layout = generated_chart
    showlegends = [trace.get('showlegend') for trace in data]
    expected_showlegends = [True, True]
    assert showlegends == expected_showlegends, "incorrect showlegend for one or multiple classes"

# test 'x' field for each class


def test_x(generated_chart):
    data, layout = generated_chart
    xs = [trace.get('x') for trace in data]
    expected_xs = [['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                   ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']]
    assert xs == expected_xs, "incorrect 'x' values for one or multiple classes"

# test 'y' field for each class


def test_y(generated_chart):
    data, layout = generated_chart
    ys = [trace.get('y') for trace in data]
    expected_ys = [[5.006, 5.936, 6.588], [1.464, 4.26, 5.552]]
    np.testing.assert_allclose(ys, expected_ys), \
        "incorrect 'y' values for one or multiple classes"

# test 'xaxis' field for each class


def test_xaxis(generated_chart):
    data, layout = generated_chart
    xaxes = [trace.get('xaxis') for trace in data]
    expected_xaxes = ['x', 'x']
    assert xaxes == expected_xaxes, "incorrect 'xaxis' values for one or multiple classes"

# test 'yaxis' field for each class


def test_yaxis(generated_chart):
    data, layout = generated_chart
    yaxes = [trace.get('yaxis') for trace in data]
    expected_yaxes = ['y', 'y']
    assert yaxes == expected_yaxes, "incorrect 'yaxis' values for one or multiple classes"

# test 'type' field for each class


def test_type(generated_chart):
    data, layout = generated_chart
    types = [trace.get('type') for trace in data]
    expected_types = ['bar', 'bar']
    assert types == expected_types, "incorrect 'type' values for one or multiple classes"


# layout

# test 'template' field in layout
def test_template(generated_chart):
    _, layout = generated_chart
    template = layout.get('template')
    assert template is not None, "'template' field not found"
    data = template.get('data')
    assert data is not None, "'data' field not found in 'template'"
    scatter = data.get('scatter')
    assert scatter is not None, "'scatter' field not found in 'data'"
    assert scatter[0].get('type') == 'scatter', "incorrect type in 'scatter'"

# test 'xaxis' field in layout


def test_xaxis(generated_chart):
    _, layout = generated_chart
    xaxis = layout.get('xaxis')
    assert xaxis is not None, "'xaxis' field not found"
    assert xaxis.get('anchor') == 'y', "incorrect 'anchor' value in 'xaxis'"
    assert xaxis.get('domain') == [
        0.0, 1.0], "incorrect 'domain' value in 'xaxis'"
    assert xaxis.get('title') == {
        'text': 'class'}, "incorrect 'title' value in 'xaxis'"
    assert xaxis.get(
        'categoryorder') == 'trace', "incorrect 'categoryorder' value in 'xaxis'"

# test 'yaxis' field in layout


def test_yaxis(generated_chart):
    _, layout = generated_chart
    yaxis = layout.get('yaxis')
    assert yaxis is not None, "'yaxis' field not found"
    assert yaxis.get('anchor') == 'x', "incorrect 'anchor' value in 'yaxis'"
    assert yaxis.get('domain') == [
        0.0, 1.0], "incorrect 'domain' value in 'yaxis'"
    assert yaxis.get('title') == {
        'text': 'value'}, "incorrect 'title' value in 'yaxis'"
    assert yaxis.get(
        'showgrid') == True, "incorrect 'showgrid' value in 'yaxis'"
    assert yaxis.get(
        'gridcolor') == 'rgba(255,0,0,.10)', "incorrect 'gridcolor' value in 'yaxis'"
    assert yaxis.get('visible') == True, "incorrect 'visible' value in 'yaxis'"
    assert yaxis.get(
        'tickformat') == '.3f', "incorrect 'tickformat' value in 'yaxis'"

# test 'legend' field in layout


def test_legend(generated_chart):
    _, layout = generated_chart
    legend = layout.get('legend')
    assert legend is not None, "'legend' field not found"
    assert legend.get('title') == {
        'text': 'Série'}, "incorrect 'title' value in 'legend'"
    assert legend.get(
        'tracegroupgap') == 0, "incorrect 'tracegroupgap' value in 'legend'"

# test 'margin' field in layout


def test_margin(generated_chart):
    _, layout = generated_chart
    margin = layout.get('margin')
    assert margin is not None, "'margin' field not found"
    assert margin.get('t') == 30, "incorrect 't' value in 'margin'"
    assert margin.get('l') == 30, "incorrect 'l' value in 'margin'"
    assert margin.get('r') == 30, "incorrect 'r' value in 'margin'"
    assert margin.get('b') == 30, "incorrect 'b' value in 'margin'"

# test 'barmode' field in layout


def test_barmode(generated_chart):
    _, layout = generated_chart
    assert layout.get('barmode') == 'stack', "incorrect 'barmode' value "
