from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from plotly import graph_objects as go


import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors



# Funnel

import pdb;pdb.set_trace()

df = util.funel_polars()

@pytest.fixture
def get_df():
    return util.funel_polars()

@pytest.fixture
def get_arguments():
    return {
        'parameters': {
            'type': 'funnel',
            'display_legend': "AUTO",
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
                "attribute": "Stage"
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
                "attribute": "Count",
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

# Test to verify if x-axis values are correct
def test_funnel_x_values(generated_chart):
    data, layout = generated_chart
    x_values = data[0].get('x')
    expected_x_values = [13873.0, 10533.0, 5443.0, 2703.0, 908.0]  
    assert x_values == expected_x_values, "Incorrect x-axis values"

# Test to verify if y-axis values are correct
def test_funnel_y_values(generated_chart):
    data, layout = generated_chart
    y_values = data[0].get('y')
    expected_y_values = ['Website visit', 'Downloads', 'Potential customers', 'Invoice sent', 'Closed deals']  
    assert y_values == expected_y_values, "Incorrect y-axis values"

# Test to verify if the 'textinfo' field is correctly set in the first data item
def test_funnel_textinfo(generated_chart):
    data, layout = generated_chart
    textinfo = data[0].get('textinfo')
    expected_textinfo = 'value+percent initial'  
    assert textinfo == expected_textinfo, "Incorrect 'textinfo' field value"

# Layout tests
# Test to verify if the layout template is correctly set
def test_funnel_layout_template(generated_chart):
    data, layout = generated_chart
    layout_template = layout.get('template')
    expected_template = {'data': {'scatter': [{'type': 'scatter'}]}}  
    assert layout_template == expected_template, "Incorrect layout template"

# Test to verify the x-axis (xaxis) settings in the layout
def test_funnel_layout_xaxis(generated_chart):
    data, layout = generated_chart
    layout_xaxis = layout.get('xaxis')
    expected_xaxis = {'categoryorder': 'trace'}  
    assert layout_xaxis == expected_xaxis, "Incorrect x-axis (xaxis) settings in the layout"
