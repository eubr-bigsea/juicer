from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors



# Bar

import pdb;pdb.set_trace()

df = util.iris_polars()

@pytest.fixture
def get_df():
    return util.iris_polars()

@pytest.fixture
def get_arguments():
    return {
    'parameters': {
        'type': 'bar',
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
            "#245668",
            "#0f7279",
            "#0d8f81",
            "#39ab7e",
            "#6ec574",
            "#a9dc67",
            "#edef5d"
        ],
        "y": [{
            "attribute": "sepallength",
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


# test 'alignmentgroup' field
def test_alignmentgroup(generated_chart):
    data, layout = generated_chart
    alignmentgroup = data[0].get('alignmentgroup')
    assert alignmentgroup is not None, "'alignmentgroup' not found in data object"
    assert alignmentgroup == 'True', "incorrect 'alignmentgroup' field value"

# test 'hovertemplate' field
def test_hovertemplate(generated_chart):
    data, layout = generated_chart
    hovertemplate = data[0].get('hovertemplate')
    assert hovertemplate is not None, "'hovertemplate' not found in data object"

# test 'legendgroup' field
def test_legendgroup(generated_chart):
    data, layout = generated_chart
    legendgroup = data[0].get('legendgroup')
    assert legendgroup is not None, "'legendgroup' not found in data object"
    assert legendgroup == 'aggr_0', "incorrect 'legendgroup' field value"

# test 'marker' field
def test_marker(generated_chart):
    data, layout = generated_chart
    marker = data[0].get('marker')
    assert marker is not None, "'marker' field not found in data object"

# test 'name' field
def test_name(generated_chart):
    data, layout = generated_chart
    name = data[0].get('name')
    assert name is not None, "'name' field not found in data object"
    assert name == 'aggr_0', "incorrect 'name' field value"

# test 'offsetgroup' field
def test_offsetgroup(generated_chart):
    data, layout = generated_chart
    offsetgroup = data[0].get('offsetgroup')
    assert offsetgroup is not None, "'offsetgroup' not found in data object"
    assert offsetgroup == 'aggr_0', "incorrect 'offsetgroup' field value"

# test 'orientation' field
def test_orientation(generated_chart):
    data, layout = generated_chart
    orientation = data[0].get('orientation')
    assert orientation is not None, "'orientation' field not found in data object"
    assert orientation == 'v', "' incorrect 'orientation' field value"

# test 'showlegend' field
def test_showlegend(generated_chart):
    data, layout = generated_chart
    showlegend = data[0].get('showlegend')
    assert showlegend is not None, "'showlegend' field not found in data object"
    assert showlegend == True, "incorrect 'showlegend' field value"

# test 'textposition' field
def test_textposition(generated_chart):
    data, layout = generated_chart
    textposition = data[0].get('textposition')
    assert textposition is not None, "'textposition' field not found in data object"
    assert textposition == 'auto', "incorrect value in 'textposition' field"

# test 'x' field
def test_x(generated_chart):
    data, layout = generated_chart
    x = data[0].get('x')
    assert x is not None, "'x' field not found in data object"

# test 'xaxis' field
def test_xaxis(generated_chart):
    data, layout = generated_chart
    xaxis = data[0].get('xaxis')
    assert xaxis is not None, "'xaxis' field not found in data object"
    assert xaxis == 'x', "incorrect 'xaxis' field value"

# test 'y' field
def test_y(generated_chart):
    data, layout = generated_chart
    y = data[0].get('y')
    assert y is not None, "'y' not found in data object"

# test 'yaxis' field
def test_yaxis(generated_chart):
    data, layout = generated_chart
    yaxis = data[0].get('yaxis')
    assert yaxis is not None, "'yaxis' not found in data object"
    assert yaxis == 'y', "incorrect 'yaxis' field value"

# test 'type' field
def test_type(generated_chart):
    data, layout = generated_chart
    chart_type = data[0].get('type')
    assert chart_type is not None, "'type' field not found in data object"
    assert chart_type == 'bar', "incorrect 'type' field value"