from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from plotly import graph_objects as go

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors



# parcoords
    
import pdb;pdb.set_trace()
    
df = util.iris2_polars()

@pytest.fixture
def get_df():
    return util.iris2_polars()

@pytest.fixture
def get_arguments():
    return {
        'parameters': {
        'type': 'parcoords',
        'display_legend': "HIDE",
        "x": [{
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
            "attribute": "Id"
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
            "attribute": "PetalLengthCm",
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
            "marker": None
        },{
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
            "marker": None
        },{
            "attribute": "SepalLengthCm",
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
            "marker": None
        },{
            "attribute": "SepalWidthCm",
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
            "marker": None
        }],
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
            "suffix": None
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
            "suffix": None
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


# Data tests
# Test to verify the 'dimensions' field in the data object
def test_data_dimensions(generated_chart):
    data, layout = generated_chart
    data_dimensions = data[0].get('dimensions')
    assert data_dimensions is not None, "Field 'dimensions' not found in the data object"
    expected_dimensions = [
        {'label': 'min(PetalLengthCm)', 'values': [1.3, 1.1, 1.0, 1.4, 1.2, 1.3, 1.4, 3.3, 3.5, 3.9, 3.5, 3.9, 3.3, 3.0, 4.5, 5.0, 4.8, 4.9, 4.8, 5.0]},
        {'label': 'min(PetalWidthCm)', 'values': [0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 1.0, 1.0, 1.1, 1.0, 1.2, 1.0, 1.1, 1.7, 1.5, 1.8, 1.4, 1.8, 1.8]},
        {'label': 'min(SepalLengthCm)', 'values': [4.6, 4.3, 4.6, 4.7, 4.8, 4.4, 4.6, 4.9, 5.0, 5.6, 5.5, 5.4, 5.0, 5.1, 4.9, 5.7, 5.6, 6.1, 6.0, 5.8]},
        {'label': 'min(SepalWidthCm)', 'values': [3.0, 2.9, 3.4, 3.0, 3.1, 2.3, 3.0, 2.3, 2.0, 2.2, 2.4, 2.3, 2.3, 2.5, 2.5, 2.2, 2.7, 2.6, 3.0, 2.5]}
    ]
    assert data_dimensions == expected_dimensions, "Incorrect values for 'dimensions' field"

# Test to verify the 'domain' field 
def test_data_domain(generated_chart):
    data, layout = generated_chart
    data_domain = data[0].get('domain')
    assert data_domain is not None, "Field 'domain' not found in the data object"
    expected_domain = {'x': [0.0, 1.0], 'y': [0.0, 1.0]}
    assert data_domain == expected_domain, "Incorrect values for 'domain' field"

# Test to verify the 'line' field 
def test_data_line(generated_chart):
    data, layout = generated_chart
    data_line = data[0].get('line')
    assert data_line is not None, "Field 'line' not found in the data object"
    expected_line = {'color': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 'coloraxis': 'coloraxis'}
    assert data_line == expected_line, "Incorrect values for 'line' field"

# Layout tests
# Test to verify the 'template' field 
def test_layout_template(generated_chart):
    data, layout = generated_chart
    layout_template = layout.get('template')
    assert layout_template is not None, "Field 'template' not found in the layout object"
    expected_template = {'data': {'scatter': [{'type': 'scatter'}]}}
    assert layout_template == expected_template, "Incorrect values for 'template' field"

# Test to verify the 'coloraxis' field 
def test_layout_coloraxis(generated_chart):
    data, layout = generated_chart
    layout_coloraxis = layout.get('coloraxis')
    assert layout_coloraxis is not None, "Field 'coloraxis' not found in the layout object"
    expected_coloraxis = {
        'colorbar': {'title': {'text': 'Id'}},
        'colorscale': [
            [0.0, '#245668'],
            [0.16666666666666666, '#0f7279'],
            [0.3333333333333333, '#0d8f81'],
            [0.5, '#39ab7e'],
            [0.6666666666666666, '#6ec574'],
            [0.8333333333333334, '#a9dc67'],
            [1.0, '#edef5d']
        ],
        'cmid': 2
    }
    assert layout_coloraxis == expected_coloraxis, "Incorrect values for 'coloraxis' field"

# Test to verify the 'legend' field 
def test_layout_legend(generated_chart):
    data, layout = generated_chart
    layout_legend = layout.get('legend')
    assert layout_legend is not None, "Field 'legend' not found in the layout object"
    expected_legend = {'tracegroupgap': 0}
    assert layout_legend == expected_legend, "Incorrect values for 'legend' field"

# Test to verify the 'xaxis' field 
def test_layout_xaxis(generated_chart):
    data, layout = generated_chart
    layout_xaxis = layout.get('xaxis')
    assert layout_xaxis is not None, "Field 'xaxis' not found in the layout object"
    expected_xaxis = {'categoryorder': 'trace'}
    assert layout_xaxis == expected_xaxis, "Incorrect values for 'xaxis' field"
