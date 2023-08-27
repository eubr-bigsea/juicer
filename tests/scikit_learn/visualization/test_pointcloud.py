from tests.scikit_learn import util
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from plotly import graph_objects as go


import json
import pytest
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.colors



# pointcloud
    
df = util.iris_polars()

arguments = {
    'parameters': {
        'type': 'pointcloud',
        'display_legend': "LEFT",
        "x": [{
            "binning": "EQUAL_INTERVAL",
            "bins": 30,
            "binSize": 10,
            "emptyBins": "ZEROS",
            "multiplier": 10,
            "decimal_places": 3,
            "prefix": None,
            "suffix": None,
            "label": None,
            "max_displayed": "",
            "group_others": True,
            "sorting": "Y_ASC",
            "attribute": "petalwidth",
            "displayLabel": "TESTE"
        },{
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
            "attribute": "species"
        }],
        "palette": [
            "#1b9e77",
            "#d95f02",
            "#7570b3",
            "#e7298a",
            "#66a61e",
            "#e6ab02",
            "#a6761d",
            "#666666"
        ],
        "y": [{
            "attribute": "petallength",
            "aggregation": "COUNT",
            "compute": None,
            "displayOn": "left",
            "multiplier": 10,
            "decimal_places": 3,
            "prefix": "",
            "suffix": None,
            "label": "TESTE",
            "strokeSize": 0,
            "stroke": None,
            "color": "#da1616",
            "marker": None,
            "custom_color": False,
            "line_color": "#141414",
            "enabled": True
        }],
        "x_axis": {
            "lowerBound": "1",
            "upperBound": "10",
            "logScale": True,
            "display": False,
            "displayLabel": False,
            "label": "TESTE",
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


df_pol = df.collect()
df_pandas = df_pol.to_pandas()

fig = px.scatter_3d(df_pandas, x=df_pandas['sepallength'], y=df_pandas['sepalwidth'], z=df_pandas['petal_width'],
              color=df_pandas['species'])

# Converter em JSON
fig_json = fig.to_json()
generated_chart_vis = json.loads(fig_json)
data1 = generated_chart_vis['data']
layout1 = generated_chart_vis['layout']

print(data1)
print(layout1)


type_test = data1[0]['type']



#data tests    
#teste type
def test_pointcloud_type():
    assert type_chart == type_test

