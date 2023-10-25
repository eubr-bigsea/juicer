import numpy as np
import pytest

from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from tests.scikit_learn import util
from tests.scikit_learn.fixtures import *
# Scatter


@pytest.fixture
def get_arguments():
    return {
        'parameters': {
            'type': 'scatter',
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
                "integerType": False
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
                "integerType": False
            },
            "y": [{
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
    code = '\n'.join([
        "import plotly.graph_objects as go",
        "import plotly.express as px",
        "import json",
        instance.generate_code(), ])
    result = util.execute(code, vis_globals)
    generated_chart = result.get('d')
    data = generated_chart['data']
    layout = generated_chart['layout']
    #
    return data, layout


def test_hovertemplate(generated_chart):
    data, layout = generated_chart
    hovertemplates = [trace.get('hovertemplate') for trace in data]
    expected_hovertemplates = [
        'class=%{x}<br>avg(petallength)=%{y}<br>petallength=%{marker.size}<extra></extra>',
        'class=%{x}<br>avg(petallength)=%{y}<br>petallength=%{marker.size}<extra></extra>',
        'class=%{x}<br>avg(petallength)=%{y}<br>petallength=%{marker.size}<extra></extra>'
    ]
    assert hovertemplates == expected_hovertemplates, "incorrect hovertemplate for one or multiple classes"

# test 'legendgroup' field for each class


def test_legendgroup(generated_chart):
    data, layout = generated_chart
    legendgroups = [trace.get('legendgroup') for trace in data]
    expected_legendgroups = ['Iris-setosa',
                             'Iris-versicolor', 'Iris-virginica']
    assert legendgroups == expected_legendgroups, "incorrect legendgroup for one or multiple classes"

# test 'marker' field for each class


def test_marker(generated_chart):
    data, layout = generated_chart
    markers = [trace.get('marker') for trace in data]
    for marker in markers:
        assert marker is not None, "'marker' field not found for one or multiple classes"
        assert 'color' in marker, "'color' field not found in 'marker'"
        assert 'size' in marker, "'size' field not found in 'marker'"
        assert 'sizemode' in marker, "'sizemode' field not found in 'marker'"
        assert 'sizeref' in marker, "'sizeref' field not found in 'marker'"
        assert 'symbol' in marker, "'symbol' field not found in 'marker'"

# test 'mode' field for each class


def test_mode(generated_chart):
    data, layout = generated_chart
    modes = [trace.get('mode') for trace in data]
    expected_modes = ['markers', 'markers', 'markers']
    assert modes == expected_modes, "incorrect mode for one or multiple classes"

# test 'name' field for each class


def test_name(generated_chart):
    data, layout = generated_chart
    names = [trace.get('name') for trace in data]
    expected_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    assert names == expected_names, "incorrect name for one or multiple classes"

# test 'orientation' field for each class


def test_orientation(generated_chart):
    data, layout = generated_chart
    orientations = [trace.get('orientation') for trace in data]
    expected_orientations = ['v', 'v', 'v']
    assert orientations == expected_orientations, "incorrect orientation for one or multiple classes"

# test 'showlegend' field for each class


def test_showlegend(generated_chart):
    data, layout = generated_chart
    showlegends = [trace.get('showlegend') for trace in data]
    expected_showlegends = [True, True, True]
    assert showlegends == expected_showlegends, "incorrect showlegend value for one or multiple classes "

# test 'x' field for each class


def test_x(generated_chart):
    data, layout = generated_chart
    xs = [trace.get('x') for trace in data]
    expected_xs = [['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa'],
                   ['Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',
                       'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor'],
                   ['Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica']]
    assert xs == expected_xs, "incorrect 'x' values for one or multiple classes  "

# test 'y' field for each class


def test_y(generated_chart, get_df):
    data, layout = generated_chart

    expected_ys = (get_df.groupby(['class', 'petallength'])
                   .agg(pl.avg('petallength').alias('avg_pl')).collect())
    # import pdb; pdb.set_trace()
    ys = [trace.get('y') for trace in data]
    expected_ys = [
        [1.5, 1.0, 1.4000000000000001, 1.9, 1.7, 1.5999999999999999, 1.1,
            1.2, 1.3],
        [4.5, 3.6, 4.8, 3.7, 4.9, 4.0, 3.8, 5.1, 3.0, 4.6, 3.9, 4.2, 4.4,
            4.7, 3.3, 3.5, 4.1, 4.3, 5.0],
        [6.0, 6.6, 6.3, 6.4, 5.1000000000000005, 5.8, 4.5, 6.9, 5.7,
         4.9, 5.6000000000000005, 5.3, 5.0, 5.4, 5.2, 5.9, 6.099999999999999,
         5.5, 6.7, 4.8]]
    for a, b in zip(ys, expected_ys):
        np.testing.assert_allclose(a, b), \
            "incorrect 'y' values for one or multiple classes"

# test 'xaxis' field for each class


def test_xaxis(generated_chart):
    data, layout = generated_chart
    xaxes = [trace.get('xaxis') for trace in data]
    expected_xaxes = ['x', 'x', 'x']
    assert xaxes == expected_xaxes, "incorrect 'xaxis' values for one or multiple classes  "

# test 'yaxis' field for each class


def test_yaxis(generated_chart):
    data, layout = generated_chart
    yaxes = [trace.get('yaxis') for trace in data]
    expected_yaxes = ['y', 'y', 'y']
    assert yaxes == expected_yaxes, "incorrect 'yaxis' values for one or multiple classes  "

# test 'type' field for each class


def test_type(generated_chart):
    data, layout = generated_chart
    types = [trace.get('type') for trace in data]
    expected_types = ['scatter', 'scatter', 'scatter']
    assert types == expected_types, "incorrect 'type' values for one or multiple classes  "


# layout

# test 'template' field in layout
def test_template(generated_chart):
    data, layout = generated_chart
    template = layout.get('template')
    assert template is not None, "'template' field not found in layout"
    assert 'data' in template, "'data' field not found in 'template'"
    assert 'scatter' in template['data'], "'scatter' field not found in 'data' of 'template' "

# test 'xaxis' field in layout


def test_xaxis(generated_chart):
    data, layout = generated_chart
    xaxis = layout.get('xaxis')
    assert xaxis is not None, "'xaxis' field not found in layout"
    assert 'anchor' in xaxis, "'anchor' field not found in 'xaxis'"
    assert 'domain' in xaxis, "'domain' field not found in 'xaxis'"
    assert 'title' in xaxis, "'title' field not found in 'xaxis'"
    assert 'text' in xaxis['title'], "'text' field not found in 'title' of 'xaxis' "
    assert 'categoryorder' in xaxis, "'categoryorder' field not found in 'xaxis'"

# test 'yaxis' field in layout


def test_yaxis(generated_chart):
    data, layout = generated_chart
    yaxis = layout.get('yaxis')
    assert yaxis is not None, "'yaxis' field not found in layout"
    assert 'anchor' in yaxis, "'anchor' field not found in 'yaxis'"
    assert 'domain' in yaxis, "'domain' field not found in 'yaxis'"
    assert 'title' in yaxis, "'title' field not found in 'yaxis'"
    assert 'text' in yaxis['title'], "'text' field not found in 'title' of 'yaxis' "
    assert 'showgrid' in yaxis, "'showgrid' field not found in 'yaxis'"
    assert 'gridcolor' in yaxis, "'gridcolor' field not found in 'yaxis'"
    assert 'visible' in yaxis, "'visible' field not found in 'yaxis'"
    assert 'tickformat' in yaxis, "'tickformat' field not found in 'yaxis'"

# test 'legend' field in layout


def test_legend(generated_chart):
    data, layout = generated_chart
    legend = layout.get('legend')
    assert legend is not None, "'legend' field not found in layout"
    assert 'title' in legend, "'title' field not found in 'legend'"
    assert 'text' in legend['title'], "'text' field not found in 'title' of 'legend' "
    assert 'tracegroupgap' in legend, "'tracegroupgap' field not found in 'legend'"
    assert 'itemsizing' in legend, "'itemsizing' field not found in 'legend'"

# test 'margin' field in layout


def test_margin(generated_chart):
    data, layout = generated_chart
    margin = layout.get('margin')
    assert margin is not None, "'margin' field not found in layout"
    assert 't' in margin, "'t' field not found in 'margin'"
    assert 'l' in margin, "'l' field not found in 'margin'"
    assert 'r' in margin, "'r' field not found in 'margin'"
    assert 'b' in margin, "'b' field not found in 'margin'"
