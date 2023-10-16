from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors



# DensityHeatmap

import pdb;pdb.set_trace()

df = util.iris_polars()

@pytest.fixture
def get_df():
    return util.iris_polars()

@pytest.fixture
def get_arguments():
    return {
        'parameters': {
            'type': 'histogram2d',
            'display_legend': "AUTO",
            "x": [{
                "binning": "EQUAL_INTERVAL",
                "bins": 20,
                "binSize": 10,
                "emptyBins": "ZEROS",
                "decimal_places": 2,
                "group_others": True,
                "sorting": "NATURAL",
                "attribute": "petalwidth"
            }],
            "color_scale": [
                "#245668",
                "#0f7279",
                "#0d8f81",
                "#39ab7e",
                "#6ec574",
                "#a9dc67",
                "#edef5d"
            ],
            "y": [{
                "attribute": "petallength",
                "aggregation": "MIN",
                "displayOn": "left",
                "decimal_places": 2,
                "strokeSize": 0,
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
            "task_id": "0"
        },
        'named_inputs': {
            'input data': "iris",
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
#
def emit_event(*args, **kwargs):
    print(args, kwargs)

@pytest.fixture
def generated_chart(get_arguments, get_df):
    instance = VisualizationOperation(**get_arguments)
    vis_globals = dict(iris=get_df, emit_event=emit_event)
    code ='\n'.join( ["import plotly.graph_objects as go","import plotly.express as px","import json",instance.generate_code(),])
    result = util.execute(code, vis_globals)
    generated_chart = result.get('d')
    data = generated_chart['data']
    layout = generated_chart['layout']
    print(data)
    return data,layout


#funções de teste 

# Test to verify the 'coloraxis' field
def test_coloraxis(generated_chart):
    data, _ = generated_chart
    coloraxes = [trace.get('coloraxis') for trace in data]
    expected_coloraxes = ['coloraxis', None, None]
    assert coloraxes == expected_coloraxes, "Incorrect 'coloraxis' value for one or more series"

# Test to verify the 'hovertemplate' field for each series
def test_hovertemplate(generated_chart):
    data, _ = generated_chart
    hovertemplates = [trace.get('hovertemplate') for trace in data]
    expected_hovertemplates = [
        'petalwidth=%{x}<br>min(petallength)=%{y}<br>count=%{z}<extra></extra>',
        'petalwidth=%{x}<extra></extra>',
        'min(petallength)=%{y}<extra></extra>'
    ]
    assert hovertemplates == expected_hovertemplates, "Incorrect hovertemplate for one or more series"

# Test to verify the 'notched' field for 'box' type series
def test_notched(generated_chart):
    data, _ = generated_chart
    box_notched = [trace.get('notched') for trace in data if trace.get('type') == 'box']
    expected_notched = [True]
    assert box_notched == expected_notched, "Incorrect 'notched' value for 'box' type series"

# Test to verify the 'scalegroup' field for 'violin' type series
def test_scalegroup(generated_chart):
    data, _ = generated_chart
    violin_scalegroup = [trace.get('scalegroup') for trace in data if trace.get('type') == 'violin']
    expected_scalegroup = ['y']
    assert violin_scalegroup == expected_scalegroup, "Incorrect 'scalegroup' value for 'violin' type series"

# Test to verify the 'x' field for each series
def test_x(generated_chart):
    data, _ = generated_chart
    xs = [trace.get('x') for trace in data]
    expected_xs = [
        [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        None
    ]
    assert xs == expected_xs, "Incorrect 'x' values for one or more series"

# Test to verify the 'xaxis' field for each series
def test_xaxis(generated_chart):
    data, _ = generated_chart
    xaxes = [trace.get('xaxis') for trace in data]
    expected_xaxes = ['x', 'x3', 'x2']
    assert xaxes == expected_xaxes, "Incorrect 'xaxis' values for one or more series"

# Test to verify the 'xbingroup' field for 'histogram2d' type series
def test_xbingroup(generated_chart):
    data, _ = generated_chart
    histogram2d_xbingroup = [trace.get('xbingroup') for trace in data if trace.get('type') == 'histogram2d']
    expected_xbingroup = ['x']
    assert histogram2d_xbingroup == expected_xbingroup, "Incorrect 'xbingroup' value for 'histogram2d' type series"

# Test to verify the 'y' field for each series
def test_y(generated_chart):
    data, _ = generated_chart
    ys = [trace.get('y') for trace in data]
    expected_ys = [
        [1.0, 1.3, 1.3, 1.7, 1.6, 3.3, 3.0, 3.9, 3.6, 4.2, 4.5, 4.5, 4.8, 4.9, 5.4, 5.6, 5.1, 5.1],
        None,
        [1.0, 1.3, 1.3, 1.7, 1.6, 3.3, 3.0, 3.9, 3.6, 4.2, 4.5, 4.5, 4.8, 4.9, 5.4, 5.6, 5.1, 5.1]
    ]
    assert ys == expected_ys, "Incorrect 'y' values for one or more series"

# Test to verify the 'yaxis' field for each series
def test_yaxis(generated_chart):
    data, _ = generated_chart
    yaxes = [trace.get('yaxis') for trace in data]
    expected_yaxes = ['y', 'y3', 'y2']
    assert yaxes == expected_yaxes, "Incorrect 'yaxis' values for one or more series"

# Test to verify the 'ybingroup' field for 'histogram2d' type series
def test_ybingroup(generated_chart):
    data, _ = generated_chart
    histogram2d_ybingroup = [trace.get('ybingroup') for trace in data if trace.get('type') == 'histogram2d']
    expected_ybingroup = ['y']
    assert histogram2d_ybingroup == expected_ybingroup, "Incorrect 'ybingroup' value for 'histogram2d' type series"

# Test to verify the 'type' field for each series
def test_type(generated_chart):
    data, _ = generated_chart
    types = [trace.get('type') for trace in data]
    expected_types = ['histogram2d', 'box', 'violin']
    assert types == expected_types, "Incorrect 'type' values for one or more series"
