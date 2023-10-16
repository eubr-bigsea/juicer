from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation

import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors

# Error tests

import pdb;pdb.set_trace()
    
df = util.titanic_polars()

@pytest.fixture
def get_df():
    return util.titanic_polars()

@pytest.fixture
def get_arguments():
    return {
        'parameters': {
        'type': 'sunburst',
        'display_legend': "HIDE",
        "x": [{
            "binning": None,
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
            "attribute": "pclass"
        }],
        "color_scale": [
            "#000000",
            "#e60000",
            "#e6d200",
            "#ffffff",
            "#a0c8ff"
        ],
        "y": [{
            "attribute": "*",
            "aggregation": "COUNT",
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
            "enabled": True
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

def generated_chart(get_arguments, get_df):
    instance = VisualizationOperation(**get_arguments)
    vis_globals = dict(iris=get_df, emit_event=emit_event)
    code ='\n'.join( ["import plotly.graph_objects as go","import plotly.express as px","import json",instance.generate_code(),])
    result = util.execute(code, vis_globals)
    generated_chart_code = result.get('d')
    data = generated_chart_code['data']
    layout = generated_chart_code['layout']
    print(data)
    return data,layout


    
def test_missing_chart_type(get_arguments, get_df):
    # Removes the chart type from the arguments
    arguments = get_arguments
    del arguments['parameters']['type']

    # Tries to generate the chart with missing parameters
    with pytest.raises(ValueError) as ex:
        result = generated_chart(arguments, get_df)
    assert "Missing required parameter: type" in str(ex.value)

def test_missing_input_data(get_arguments, get_df):
    # Simulates absence of data by setting the 'get_df' variable to None
    get_df = None

    # Sets the input data to None to test the case where the data is not provided
    arguments = get_arguments
    arguments['named_inputs']['input data'] = get_df

    # Tries to generate the chart with missing input data
    with pytest.raises(Exception) as ex:
        result = generated_chart(arguments, get_df)

    # Verifies if the expected error message was raised, error caused by missing df
    assert str(ex.value) == "'NoneType' object has no attribute 'clone'"

def test_missing_x(get_arguments, get_df):
    # Removes the x parameter from the arguments
    arguments = get_arguments.copy()
    del arguments['parameters']['x']

    # Tries to generate the chart with missing parameters, x values from the dataset
    with pytest.raises(ValueError) as ex:
        result, _ = generated_chart(arguments, get_df)
    assert "Missing required parameter: x" in str(ex.value)

def test_missing_y(get_arguments, get_df):
    # Removes the y parameter from the arguments
    arguments = get_arguments.copy()
    del arguments['parameters']['y']

    # Tries to generate the chart with missing parameters, y values from the dataset
    with pytest.raises(ValueError) as ex:
        result, _ = generated_chart(arguments, get_df)
    assert "Missing required parameter: y" in str(ex.value)

def test_missing_x_axis(get_arguments, get_df):
    # Removes the x axis parameter from the arguments
    arguments = get_arguments.copy()
    del arguments['parameters']['x_axis']

    with pytest.raises(ValueError) as ex:
        result, _ = generated_chart(arguments, get_df)
    assert "Missing required parameter: x_axis" in str(ex.value)

def test_missing_y_axis(get_arguments, get_df):
    # Removes the y axis parameter from the arguments
    arguments = get_arguments.copy()
    del arguments['parameters']['y_axis']

    with pytest.raises(ValueError) as ex:
        result, _ = generated_chart(arguments, get_df)
    assert "Missing required parameter: y_axis" in str(ex.value)

def test_missing_display_legend(get_arguments, get_df):
    # Removes the display_legend parameter from the arguments
    arguments = get_arguments.copy()
    del arguments['parameters']['display_legend']

    # Tries to generate the chart with missing parameters
    with pytest.raises(ValueError) as ex:
        result, _ = generated_chart(arguments, get_df)
    assert "Missing required parameter: display_legend" in str(ex.value)
