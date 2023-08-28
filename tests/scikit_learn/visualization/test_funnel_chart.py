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
    
df = util.funel_polars()

arguments = {
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


    

instance = VisualizationOperation(**arguments)
    
def emit_event(*args, **kwargs):
    print(args, kwargs)

vis_globals = dict(iris=df, emit_event=emit_event)
code ='\n'.join( ["import plotly.graph_objects as go","import plotly.express as px","import json",instance.generate_code(),])
result = util.execute(code, 
                          vis_globals)

# Use result.get('d') to get the Python dict containing the chart
generated_chart = result.get('d')
import pdb;pdb.set_trace()

data = generated_chart['data']
layout = generated_chart['layout']

print(data)
print(layout)
print(data[0]['x'])

type_chart = data[0]['type']
x_chart = data[0]['x']
y_chart = data[0]['y']
text_info_chart = data[0]['textinfo']

df_pol = df.collect()
df_pandas = df_pol.to_pandas()

fig = go.Figure(go.Funnel(
    y = df_pandas['Stage'],
    x = df_pandas['Count'],
    textposition = "inside",
    textinfo = "value+percent initial",
    opacity = 0.65, marker = {"color": ["deepskyblue", "lightsalmon", "tan", "teal", "silver"],
    "line": {"width": [4, 2, 2, 3, 1, 1], "color": ["wheat", "wheat", "blue", "wheat", "wheat"]}},
    connector = {"line": {"color": "royalblue", "dash": "dot", "width": 3}})
    )

# Converter em JSON
fig_json = fig.to_json()
generated_chart_vis = json.loads(fig_json)
data1 = generated_chart_vis['data']
layout1 = generated_chart_vis['layout']

print(data1)
print(layout1)
print(data1[0]['x'])
print(data1[0]['type'])
print(data1[0]['textinfo'])


type_test = data1[0]['type']
x_test = data1[0]['x']
y_test = data1[0]['y']
text_info_chart = data1[0]['textinfo']


#data tests    
#teste type
def test_funnel_type():
    assert type_chart == type_test

#teste eixo x
def test_funnel_eixo_x():
    assert x_chart == x_test

#teste eixo y
def test_funnel_eixo_y():
    assert y_chart == y_test

#teste info
def test_funnel_info():
    assert data[0]['textinfo'] == data1[0]['textinfo']

